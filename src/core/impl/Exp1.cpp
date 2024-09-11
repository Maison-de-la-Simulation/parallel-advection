#include "IAdvectorX.h"
#include "advectors.h"
#include <cstddef>
#include <experimental/mdspan>
#include <memory>
#include <type_traits>

using mdspan_t =
    std::experimental::mdspan<double, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;

// this is streamY + straddlel malloc + twoDimwg
//  TODO: implement percent_in_global_mem values

// constexpr size_t MAX_NX_ALLOC = 6144;   // A100
constexpr size_t MAX_NX_ALLOC = 64;
constexpr size_t MAX_NY = 128;

template<typename RealType>
struct StraddledBuffer {
    template<typename Tp>
    using ptr_t = std::shared_ptr<Tp>;

    using globAcc2D =
        typename sycl::accessor<RealType, 2,
                                sycl::access::mode::discard_read_write,
                                sycl::target::global_buffer>;
    using localAcc2D = typename sycl::local_accessor<RealType, 2>;

    ptr_t<globAcc2D> m_global_acc;
    ptr_t<localAcc2D> m_local_acc;
    size_t localNy, localNx; //TODO: careful if localNx is different than MAX_NX_ALLOC, we might want that ? For now we stay simple with only localNx == MAX_NX_ALLOC
    size_t globalNy, globalNx; //size of local buffer

    StraddledBuffer() = delete;
    StraddledBuffer(const globAcc2D &globalAcc, const localAcc2D &localAcc)
        : globalNy(globalAcc.get_range().get(0)),
          globalNx(globalAcc.get_range().get(1)),
          localNy(localAcc.get_range().get(0)),
          localNx(localAcc.get_range().get(1)) {
        m_global_acc = std::make_unique<globAcc2D>(globalAcc);
        m_local_acc  = std::make_unique<localAcc2D>(localAcc);
    }

    RealType& operator()(size_t iy, size_t ix) {
        if(ix < localNx)
            // return localMdspan()(iy % localNy, ix);
            return (*m_local_acc)[iy % localNy][ix];
        else
            // return globalMdspan()(iy, ix - localNx);
            return (*m_global_acc)[iy][ix - localNx];
    }
    
    const RealType operator()(size_t iy, size_t ix) const{
        if(ix < localNx)
            return localMdspan()(iy % localNy, ix);
        else
            return globalMdspan()(iy, ix - localNx);
    }

private:
    mdspan_t localMdspan(){return mdspan_t(m_local_acc->get_pointer(), localNy, localNx);}
    const mdspan_t localMdspan() const {return mdspan_t(m_local_acc->get_pointer(), localNy, localNx);}

    mdspan_t globalMdspan(){return mdspan_t(m_global_acc->get_pointer(), globalNy, globalNx);}
    const mdspan_t globalMdspan() const {return mdspan_t(m_global_acc->get_pointer(), globalNy, globalNx);}
    // StraddledBuffer(const float percentage_in_local_mem, const size_t ny,
    //                 const size_t nx, const size_t ny1, sycl::handler &cgh) {
    // }


    // ctor(percentage alloc, ny, nx, ny1, cgh_for_local_accessor)

    // index_limit
    //  operator[i] doing the if i > index_limit return global accessor else
    //  return local_accessor
};

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::actual_advection(sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset,
                             const size_t &nx_rest_to_malloc) {

    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_y = params.wg_size_y;
    auto const wg_size_x = params.wg_size_x;

    /* ny must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_y != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_y");
    }
    // if (wg_size_y * nx > 6144) {
    //     std::cout << "wg_size_y = " << wg_size_y << ", nx = " << nx
    //               << std::endl;
    //     throw std::invalid_argument(
    //         "wg_size_y*nx must be < to 6144 (shared memory limit)");
    // }

    const sycl::range nb_wg{ny_batch_size / wg_size_y, 1, ny1};
    const sycl::range wg_size{params.wg_size_y, params.wg_size_x, 1};

    sycl::buffer<double, 2> buff_rest_nx(sycl::range{ny, nx_rest_to_malloc},
                                         sycl::no_init);

    /*What about two kernels the first one fills the overslice as write only
memory and the second one fills the local accessor and solves the advection?*/

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

         StraddledBuffer<double>::globAcc2D overslice_ftmp =
            buff_rest_nx.get_access<sycl::access::mode::discard_read_write>(
                cgh);

        /* We use a 2D local accessor here */
        auto local_malloc_size = nx > MAX_NX_ALLOC ? MAX_NX_ALLOC : nx;
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, local_malloc_size), cgh, sycl::no_init);


        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            // if y1 > 1 //if we have a stide, we transpose, else we copy
            mdspan_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
            StraddledBuffer<double> BUFF(overslice_ftmp, slice_ftmp);

            /* Copy kernel*/
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;


                    BUFF(iy, ix) = fdist_view(iy, ix, iy1);
                    // if (ix < MAX_NX_ALLOC) {
                    //     // slice_ftmp[local_ny][ix] = fdist[iy][ix][iy1];
                    //     (*BUFF.m_local_acc)[local_ny][ix] = fdist_view(iy, ix, iy1);
                    //     // slice_ftmp[local_ny][ix] = fdist_view(iy, ix, iy1);
                    // } else {
                    //     (*BUFF.m_global_acc)[iy][ix - MAX_NX_ALLOC] = fdist_view(iy, ix, iy1);
                    //     // overslice_ftmp[iy][ix - MAX_NX_ALLOC] = fdist_view(iy, ix, iy1);
                    //     // fdist[iy][ix][iy1];
                    // }
                });   // barrier

            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    double const xFootCoord = displ(ix, iy, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    fdist_view(iy, ix, iy1) = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        fdist_view(iy, ix, iy1) +=
                                coef[k] * BUFF(iy, idx_ipos1);

                        // if (idx_ipos1 < MAX_NX_ALLOC) {
                        //     fdist_view(iy, ix, iy1) +=
                        //         coef[k] * (*BUFF.m_local_acc)[local_ny][idx_ipos1];
                        //         // coef[k] * slice_ftmp[local_ny][idx_ipos1];
                        // } else {
                        //     fdist_view(iy, ix, iy1) +=
                        //         coef[k] *
                        //         (*BUFF.m_global_acc)[iy][idx_ipos1 - MAX_NX_ALLOC];
                        //         // overslice_ftmp[iy][idx_ipos1 - MAX_NX_ALLOC];
                        // }

                    }
                });   // end parallel_for_work_item --> Implicit barrier
        });           // end parallel_for_work_group
    });               // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;

    auto rest_malloc = nx <= MAX_NX_ALLOC ? 0 : nx - MAX_NX_ALLOC;

    // On A100 it breaks when ny (the first dimension) is >= 65536.
    if (ny < MAX_NY) {
        /* If limit not exceeded we return a classical Hierarchical advector */
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        double div = static_cast<double>(ny) / static_cast<double>(MAX_NY);
        auto floor_div = std::floor(div);
        auto is_int = div == floor_div;
        auto n_batch = is_int ? div : floor_div + 1;

        for (int i_batch = 0; i_batch < n_batch - 1;
             ++i_batch) {   // can we parallel_for this on multiple GPUs?
                            // multiple queues ? or other CUDA streams?

            size_t ny_offset = (i_batch * MAX_NY);

            actual_advection(Q, buff_fdistrib, params, MAX_NY, ny_offset,
                             rest_malloc)
                .wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
        auto const ny_size =
            is_int ? MAX_NY : (ny % MAX_NY);   // + (n_batch - 1);
        auto const ny_offset = MAX_NY * (n_batch - 1);

        // return the last advection with the rest
        return actual_advection(Q, buff_fdistrib, params, ny_size, ny_offset,
                                rest_malloc);
    }
}