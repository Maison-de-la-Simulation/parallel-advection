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
//     typename sycl::accessor<real_t, 3,
//     sycl::access::mode::discard_read_write,
//                             sycl::target::global_buffer>;

using localAcc2D = typename sycl::local_accessor<real_t, 2>;

template <typename RealType> struct HybridMemBuffer {

    mdspan2d_t m_local_mdpsan;
    mdspan3d_t m_global_mdpsan;

    HybridMemBuffer() = delete;
    // HybridMemBuffer(const globAcc2D &globalAcc, const localAcc2D &localAcc)
    //     : m_local_mdpsan(localAcc.get_pointer(), localAcc.get_range().get(0),
    //                      localAcc.get_range().get(1)),
    //       m_global_mdpsan(globalAcc.get_pointer(),
    //       globalAcc.get_range().get(0),
    //                       globalAcc.get_range().get(1), glo) {}

    HybridMemBuffer(RealType *globalPtr, const size_t globalNy,
                 const size_t globalNx, const size_t globalNy1,
                 const localAcc2D &localAcc)
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
AdvX::Exp1::actual_advection(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver, const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    auto const wg_size_0 = 1;//solver.p.loc_wg_size_0;
    auto const wg_size_1 = solver.p.pref_wg_size;

    /* n0 must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_0 != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_0");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_0, 1, n2};
    const sycl::range wg_size{wg_size_0, wg_size_1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        /* We use a 2D local accessor here */
        // auto local_malloc_size = n1 > MAX_NX_ALLOC ? MAX_NX_ALLOC : n1;
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_0, local_alloc_size_), cgh, sycl::no_init);

        auto ptr_global = global_vertical_buffer_;
        auto n0_rest_malloc = n0_rest_malloc_;

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            // if y1 > 1 //if we have a stide, we transpose, else we copy

            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                    HybridMemBuffer<double> scratch(ptr_global, n0, n0_rest_malloc,
                                                 n2, slice_ftmp);

                    const int i1 = it.get_local_id(1);
                    const int i2 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int i0 =
                        wg_size_0 * g.get_group_id(0) + ny_offset + local_ny;

                    auto slice = std::experimental::submdspan(
                        fdist_view, i0, std::experimental::full_extent, i2);

                    scratch(i0, i1, i2) = solver(slice, i0, i1, i2);
                });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            g.parallel_for_work_item(
                sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                    HybridMemBuffer<double> scratch(ptr_global, n0, n0_rest_malloc,
                                                 n2, slice_ftmp);

                    const int i1 = it.get_local_id(1);
                    const int i2 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int i0 =
                        wg_size_0 * g.get_group_id(0) + ny_offset + local_ny;

                    fdist_view(i0, i1, i2) = scratch(i0, i1, i2);
                });   // barrier
        });           // end parallel_for_work_group
    });               // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {

    // can be parallel on multiple queues ? or other CUDA streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {
        size_t ny_offset = (i_batch * MAX_NY_BATCHS_);
        actual_advection(Q, fdist_dev, solver, MAX_NY_BATCHS_, ny_offset)
            .wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, solver, last_n0_size_,
                            last_n0_offset_);
}



// // =============================================================================
// class StraddledMalloc : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;
//     sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
//                                  const Solver &solver,
//                                  const size_t &ny_batch_size,
//                                  const size_t &ny_offset);

//     /* Max number of batch submitted */
//     static constexpr size_t MAX_NY_BATCHS_ = 65535;
//     /* Max number of elements in the local accessor */
//     // static constexpr size_t MAX_NX_ALLOC = 64;

//     sycl::queue q_;
//     double *global_vertical_buffer_;
//     size_t n_batch_;
//     size_t last_n0_size_;
//     size_t last_n0_offset_;
//     size_t n0_rest_malloc_;
//     size_t local_alloc_size_;

//     void init_batchs(const Solver &s) {
//         /* Compute number of batchs */
//         float div =
//             static_cast<float>(s.p.n0) / static_cast<float>(MAX_NY_BATCHS_);
//         auto floor_div = std::floor(div);
//         auto div_is_int = div == floor_div;
//         n_batch_ = div_is_int ? div : floor_div + 1;

//         last_n0_size_ = div_is_int ? MAX_NY_BATCHS_ : (s.p.n0 % MAX_NY_BATCHS_);
//         last_n0_offset_ = MAX_NY_BATCHS_ * (n_batch_ - 1);
//     }

//   public:
//     sycl::event operator()(sycl::queue &Q, double *fdist_dev,
//                            const Solver &solver) override;

//     StraddledMalloc() = delete;

//     StraddledMalloc(const Solver &s, const sycl::queue &q) : q_(q) {
//         init_batchs(s);

//         local_alloc_size_ = std::floor(s.p.percent_loc * s.p.n1);
//         n0_rest_malloc_ = s.p.n1 - local_alloc_size_;

//         std::cout << "Local alloc size: " << local_alloc_size_ << std::endl;
//         std::cout << "Global alloc size: " << n0_rest_malloc_ << std::endl;

//         if (n0_rest_malloc_ > 0) {
//             // TODO: don't allocate full n0, only the current batch_size_ny size
//             global_vertical_buffer_ = sycl::malloc_device<double>(
//                 s.p.n0 * n0_rest_malloc_ * s.p.n2, q_);
//         } else {
//             global_vertical_buffer_ = nullptr;
//         }
//     }

//     ~StraddledMalloc() {
//         if (n0_rest_malloc_ > 0)
//             sycl::free(global_vertical_buffer_, q_);
//     }
// };
