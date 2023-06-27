#pragma once
#include <IAdvectorVx.h>

namespace advector {

namespace vx {

class MDRange : public IAdvectorVx {
    // copy of fdist buffer because MDRange has to be done out-of-place
    KV_double_3d m_ftmp;

  public:
    MDRange(const size_t n_fict_dim, const size_t nvx, const size_t nx)
        : m_ftmp{"ftmp", n_fict_dim, nvx, nx} {}

  public:
    void operator()(KV_double_3d &fdist, KV_double_1d &elec_field,
                    const ADVParams &params) noexcept override;
};

}   // namespace vx

}   // namespace advector