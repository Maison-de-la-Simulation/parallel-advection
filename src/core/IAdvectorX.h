#pragma once

#include <AdvectionSolver.h>
#include <sycl/sycl.hpp>


// ==========================================
// ==========================================
class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const AdvectionSolver &solver) = 0;
};

// ==========================================
// ==========================================
inline void
print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
    if (lvl == 0)
        std::cout << "--------------------------------" << std::endl;
    std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
              << r.get(2) << "}" << std::endl;
}

// // ==========================================
// // ==========================================
// [[nodiscard]] inline KernelDispatch
// init_kernel_splitting(const float p, const size_t n) {
//     KernelDispatch k_dispatch;

//     auto div =  n * p;
//     k_dispatch.k_local_ = sycl::floor(static_cast<float>(div));

//     k_dispatch.k_global_ =  n - k_dispatch.k_local_;

//     return k_dispatch;
// }
