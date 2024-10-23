#include "IAdvectorX.h"
#include "advectors.h"
#include <cstddef>

/* =================================================================
Straddledmalloc: horizontal distribution of data between local and global
memory
- Only one type of kernel
- Modulo in the accessor [] of the data
- Streaming in Y with blocks BY
- GridStride (Done by hierarchical) in BY and X dims
==================================================================== */

// using globAcc3D =
//     typename sycl::accessor<real_t, 3, sycl::access::mode::discard_read_write,
//                             sycl::target::global_buffer>;

using localAcc2D = typename sycl::local_accessor<real_t, 2>;

template <typename RealType> struct StraddledBuffer {

    mdspan2d_t m_local_mdpsan;
    mdspan3d_t m_global_mdpsan;

    StraddledBuffer() = delete;
    // StraddledBuffer(const globAcc2D &globalAcc, const localAcc2D &localAcc)
    //     : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
    //                      localAcc.get_range().get(1)),
    //       m_global_mdpsan(globalAcc.get_pointer(), globalAcc.get_range().get(0),
    //                       globalAcc.get_range().get(1), glo) {}

    StraddledBuffer(RealType *globalPtr, const size_t globalNy,
                    const size_t globalNx, const size_t globalNy1, const localAcc2D &localAcc)
        : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
                         localAcc.get_range().get(1)),
          m_global_mdpsan(globalPtr, globalNy, globalNx, globalNy1) {}

    RealType &operator()(size_t i0, size_t i1, size_t i2) const {
        auto localNx = m_local_mdpsan.extent(1);
        auto localNy = m_local_mdpsan.extent(0);

        if (i1 < localNx) {
            return m_local_mdpsan(i0 % localNy, i1);
        } else {
            return m_global_mdpsan(i0, i1 - localNx, i2);
        }
    }

    // ctor(percentage alloc, n0, n1, n2, cgh_for_local_accessor)
};

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::actual_advection(sycl::queue &Q,
                             double* fdist_dev,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_0 = params.wg_size_0;
    auto const wg_size_1 = params.wg_size_1;

    /* n0 must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_0 != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_0");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_0, 1, n2};
    const sycl::range wg_size{wg_size_0, wg_size_1, 1};

    // sycl::buffer<double, 2> buff_rest_nx(sycl::range{n0, nx_rest_malloc_},
    //                                      sycl::no_init);

    return Q.submit([&](sycl::handler &cgh) {

        /* We use a 2D local accessor here */
        auto local_malloc_size = n1 > MAX_NX_ALLOC ? MAX_NX_ALLOC : n1;
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_0, local_malloc_size), cgh, sycl::no_init);

        auto ptr_global = global_vertical_buffer_;
        auto nx_rest_malloc = nx_rest_malloc_;

        cgh.parallel_for_work_group(
            nb_wg, wg_size, [=](sycl::group<3> g) {
                // if y1 > 1 //if we have a stide, we transpose, else we copy

                /* Copy kernel*/
                g.parallel_for_work_item(
                    sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                        mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                        StraddledBuffer<double> BUFF(
                            ptr_global, n0, nx_rest_malloc, n2, slice_ftmp);

                        const int i1 = it.get_local_id(1);
                        const int i2 = g.get_group_id(2);

                        const int local_ny = it.get_local_id(0);
                        const int i0 = wg_size_0 * g.get_group_id(0) +
                                       ny_offset + local_ny;

                        BUFF(i0, i1, i2) = fdist_view(i0, i1, i2);
                    });   // barrier

                /* Solve kernel */
                g.parallel_for_work_item(
                    sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                        mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                        StraddledBuffer<double> BUFF(
                            ptr_global, n0, nx_rest_malloc, n2, slice_ftmp);

                        const int i1 = it.get_local_id(1);
                        const int i2 = g.get_group_id(2);

                        const int local_ny = it.get_local_id(0);
                        const int i0 = wg_size_0 * g.get_group_id(0) +
                                       ny_offset + local_ny;

                        double const xFootCoord = displ(i1, i0, params);

                        // index of the cell to the left of footCoord
                        const int leftNode =
                            sycl::floor((xFootCoord - minRealX) * inv_dx);

                        const double d_prev1 =
                            LAG_OFFSET +
                            inv_dx *
                                (xFootCoord - coord(leftNode, minRealX, dx));

                        auto coef = lag_basis(d_prev1);

                        const int ipos1 = leftNode - LAG_OFFSET;

                        fdist_view(i0, i1, i2) = 0.;
                        for (int k = 0; k <= LAG_ORDER; k++) {
                            int id1_ipos = (n1 + ipos1 + k) % n1;

                            fdist_view(i0, i1, i2) +=
                                coef[k] * BUFF(i0, id1_ipos, i2);
                        }
                    });   // end parallel_for_work_item --> Implicit barrier
            });           // end parallel_for_work_group
    });                   // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::operator()(sycl::queue &Q, double* fdist_dev,
                       const ADVParams &params) {

    // can be parallel on multiple queues ? or other CUDA streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {
        size_t ny_offset = (i_batch * MAX_NY_BATCHS);
        actual_advection(Q, fdist_dev, params, MAX_NY_BATCHS, ny_offset)
            .wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, params, last_ny_size_,
                            last_ny_offset_);
}