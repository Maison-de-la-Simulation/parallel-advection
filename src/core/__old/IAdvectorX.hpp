#pragma once

#include <AdvectionSolver.hpp>
#include <sycl/sycl.hpp>
#include <types.hpp>

// ==========================================
// ==========================================
class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
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
