#pragma once
#include <bkma_tools.hpp>
// class BasicRange : public IAdvectorX {
//     protected:
//       sycl::queue q_;
//       real_t *ftmp_;

//     public:
//       BasicRange(const AdvectionSolver &solver, sycl::queue q) {
//           const auto n0 = solver.params.n0;
//           const auto n1 = solver.params.n1;
//           const auto n2 = solver.params.n2;

//           ftmp_ = sycl::malloc_device<real_t>(n0 * n1 * n2, q_);
//           q_.wait();
//       }

//       ~BasicRange() {
//           sycl::free(ftmp_, q_);
//           q_.wait();
//       }

//       sycl::event operator()(sycl::queue &Q, real_t *data,
//                              const AdvectionSolver &solver) override;
//   };

template <MemorySpace MemType, class MySolver, BkmaImpl Impl>
inline std::enable_if_t<Impl == BkmaImpl::BasicRange, sycl::event>
submit_kernels(sycl::queue &Q, span3d_t data, const MySolver &solver,
               const size_t b0_size, const size_t b0_offset,
               const size_t b2_size, const size_t b2_offset,
               const size_t orig_w0, const size_t w1, const size_t orig_w2,
               WorkGroupDispatch wg_dispatch, span3d_t global_scratch) {

    static_assert(
        !(MemType == MemorySpace::Local && BkmaImpl::BasicRange == Impl),
        "BasicRange is not supported with MemorySpace::Local");

    auto n0 = data.extent(0);
    auto n1 = data.extent(1);
    auto n2 = data.extent(2);

    sycl::range r3d(n0, n1, n2);

    Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
            const int i1 = itm[1];
            const int i0 = itm[0];
            const int i2 = itm[2];

            global_scratch(i0, i1, i2) =
                solver(std::experimental::submdspan(
                           data, i0, std::experimental::full_extent, i2),
                       i0, i1, i2);
            // barrier
        });   // end parallel_for
    });       // end Q.submit
    Q.wait();
    // copy
    return Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
            const int i1 = itm[1];
            const int i0 = itm[0];
            const int i2 = itm[2];
            data(i0, i1, i2) = global_scratch(i0, i1, i2);
            // barrier
        });   // end parallel_for
    });       // end Q.submit
}
