#pragma once

#include "../kokkos_shortcut.hpp"
#include "AdvectionParams.h"
#include <IAdvector.h>

class IAdvectorVx : public IAdvector {
  public:
    virtual void operator()(KV_double_3d &fdist, KV_double_1d &elec_field,
                            const ADVParams &params) noexcept = 0;
};