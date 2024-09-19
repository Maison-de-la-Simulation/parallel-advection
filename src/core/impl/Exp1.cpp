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
    typename sycl::accessor<real_t, 2, sycl::access::mode::discard_read_write,
                            sycl::target::global_buffer>;

using localAcc2D = typename sycl::local_accessor<real_t, 2>;

template <typename RealType> struct StraddledBuffer {

    mdspan2d_t m_local_mdpsan;
    mdspan2d_t m_global_mdpsan;

    StraddledBuffer() = delete;
    StraddledBuffer(const globAcc2D &globalAcc, const localAcc2D &localAcc)
        : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
                         localAcc.get_range().get(1)),
          m_global_mdpsan(globalAcc.get_pointer(), globalAcc.get_range().get(0),
                          globalAcc.get_range().get(1)) {}

    StraddledBuffer(RealType *globalPtr, const size_t globalNy,
                    const size_t globalNx, const localAcc2D &localAcc)
        : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
                         localAcc.get_range().get(1)),
          m_global_mdpsan(globalPtr, globalNy, globalNx) {}

    RealType &operator()(size_t iy, size_t ix) const {
        auto localNx = m_local_mdpsan.extent(1);
        auto localNy = m_local_mdpsan.extent(0);

        if (ix < localNx) {
            return m_local_mdpsan(iy % localNy, ix);
        } else {
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
    auto const wg_size_x = params.wg_size_x;

    /* ny must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_y != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_y");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_y, 1, ny1};
    const sycl::range wg_size{wg_size_y, wg_size_x, 1};

    // sycl::buffer<double, 2> buff_rest_nx(sycl::range{ny, nx_rest_malloc_},
    //                                      sycl::no_init);

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        // globAcc2D overslice_ftmp =
        //     buff_rest_nx.get_access<sycl::access::mode::discard_read_write>(
        //         cgh);

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
                    StraddledBuffer<double> BUFF(global_vertical_buffer_, ny,
                                                 nx_rest_malloc_, slice_ftmp);

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    BUFF(iy, ix) = fdist_view(iy, ix, iy1);
                });   // barrier

            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    StraddledBuffer<double> BUFF(global_vertical_buffer_, ny,
                                                 nx_rest_malloc_, slice_ftmp);

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

    // can be parallel on multiple queues ? or other CUDA streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {
        size_t ny_offset = (i_batch * MAX_NY_BATCHS);
        actual_advection(Q, buff_fdistrib, params, MAX_NY_BATCHS, ny_offset)
            .wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, buff_fdistrib, params, last_ny_size_,
                            last_ny_offset_);
}