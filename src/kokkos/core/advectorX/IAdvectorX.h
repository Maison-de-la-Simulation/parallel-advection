#pragma once

#include "AdvectionParams.h"
#include <IAdvector.h>
#include "../kokkos_shortcut.hpp"

class IAdvectorX : public IAdvector {
  public:
    virtual void operator()(KV_double_3d &fdistrib,
                            const ADVParams &params) noexcept = 0;
};