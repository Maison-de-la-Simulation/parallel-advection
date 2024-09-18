#include "IAdvectorX.h"
#include "advectors.h"
#include <cstddef>
#include <experimental/mdspan>

using real_t = double;

using mdspan3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;
using mdspan2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

using globAcc2D =
    typename sycl::accessor<real_t, 2,
                            sycl::access::mode::discard_read_write,
                            sycl::target::global_buffer>;

using localAcc2D = typename sycl::local_accessor<real_t, 2>;
//TODO: implement percent_in_global_mem values

//constexpr size_t MAX_NX_ALLOC = 6144;   // A100

// constexpr size_t MAX_NX_ALLOC = 6144;
// constexpr size_t MAX_NY = 65535;

template <typename RealType> struct StraddledBuffer {

    mdspan2d_t m_local_mdpsan;
    mdspan2d_t m_global_mdpsan;

    StraddledBuffer() = delete;
    StraddledBuffer(const localAcc2D &localAcc, RealType *global_ptr,
                    size_t globalNx, size_t globalNy)
        : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
                         localAcc.get_range().get(1)),
          m_global_mdpsan(global_ptr, globalNy, globalNx) {}

    RealType &operator()(size_t iy, size_t ix) const {
        auto localNx = m_local_mdpsan.extent(1);
        auto localNy = m_local_mdpsan.extent(0);

        if (ix < localNx){
            return m_local_mdpsan(iy % localNy, ix);
        }
        else{
            return m_global_mdpsan(iy, ix - localNx);
        }
    }

    // ctor(percentage alloc, ny, nx, ny1, cgh_for_local_accessor)
};

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::actual_advection(sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_y = params.wg_size_y;

    /* ny must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_y != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_y");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_y, 1, ny1};
    const sycl::range wg_size{params.wg_size_y, params.wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* We use a 2D local accessor here */
        auto local_malloc_size = nx > MAX_NX_ALLOC ? MAX_NX_ALLOC : nx;
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, local_malloc_size), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            // if y1 > 1 //if we have a stide, we transpose, else we copy

            /* Copy kernel*/
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    StraddledBuffer<double> BUFF(
                        slice_ftmp, this->buffer_rest_nx, this->ny_,
                        this->overslice_nx_size_);

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    BUFF(iy, ix) = fdist_view(iy, ix, iy1);
                    // if (ix < MAX_NX_ALLOC) {
                    //     // slice_ftmp[local_ny][ix] = fdist[iy][ix][iy1];
                    //     (*BUFF.m_local_acc)[local_ny][ix] = fdist_view(iy,
                    //     ix, iy1);
                    //     // slice_ftmp[local_ny][ix] = fdist_view(iy, ix,
                    //     iy1);
                    // } else {
                    //     (*BUFF.m_global_acc)[iy][ix - MAX_NX_ALLOC] =
                    //     fdist_view(iy, ix, iy1);
                    //     // overslice_ftmp[iy][ix - MAX_NX_ALLOC] =
                    //     fdist_view(iy, ix, iy1);
                    //     // fdist[iy][ix][iy1];
                    // }
                });   // barrier

            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    StraddledBuffer<double> BUFF(
                        slice_ftmp, this->buffer_rest_nx, this->ny_,
                        this->overslice_nx_size_);

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
                        //         coef[k] *
                        //         (*BUFF.m_local_acc)[local_ny][idx_ipos1];
                        //         // coef[k] * slice_ftmp[local_ny][idx_ipos1];
                        // } else {
                        //     fdist_view(iy, ix, iy1) +=
                        //         coef[k] *
                        //         (*BUFF.m_global_acc)[iy][idx_ipos1 -
                        //         MAX_NX_ALLOC];
                        //         // overslice_ftmp[iy][idx_ipos1 -
                        //         // // MAX_NX_ALLOC];
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
    // auto const nx = params.nx;
    // auto const ny = params.ny;
    // auto const ny1 = params.ny1;

    // On A100 it breaks when ny (the first dimension) is >= 65536.
    // if (ny < MAX_NY) {
    //     /* If limit not exceeded we return a classical Hierarchical advector */
    //     AdvX::Hierarchical adv{};
    //     return adv(Q, buff_fdistrib, params);
    // } else {
        // double div = static_cast<double>(ny) / static_cast<double>(MAX_NY);
        // auto floor_div = std::floor(div);
        // auto is_int = div == floor_div;
        // auto n_batch = is_int ? div : floor_div + 1;

            // can we parallel_for this on multiple queues ? or CUDA streams?
        for (size_t i_batch = 0; i_batch < this->n_batch_ - 1; ++i_batch) {
            size_t ny_offset = (i_batch * this->MAX_NY_BATCH);

            actual_advection(Q, buff_fdistrib, params, this->MAX_NY_BATCH, ny_offset)
                .wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
  
        // return the last advection with the rest
        return actual_advection(Q, buff_fdistrib, params, last_batch_size_ny_,
                                last_batch_offset_ny_);
        // }
}