#pragma once
#include "../kokkos_shortcut.hpp"
#include <IAdvectorX.h>

/* Contains headers for different implementations of advector interface */
namespace advector {

namespace x {

class MDRange : public IAdvectorX {
    // copy of fdist buffer because MDRange has to be done out-of-place
    mutable KV_double_3d m_ftmp;

  public:
    MDRange(const size_t n_fict_dim, const size_t nvx, const size_t nx)
        : m_ftmp{"fdist", n_fict_dim, nvx, nx} {}

    void operator()(KV_double_3d &fdistrib,
                    const ADVParams &params) noexcept override;
};

}   // namespace x

}   // namespace advector
