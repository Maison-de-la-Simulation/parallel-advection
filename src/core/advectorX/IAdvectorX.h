#pragma once

#include "AdvectionParams.h"
#include <IAdvector.h>
#include <sycl/sycl.hpp>

class IAdvectorX : public IAdvector {
  public:
    virtual sycl::event operator()(sycl::queue &Q,
                                   sycl::buffer<double, 2> &buff_fdistrib,
                                   const ADVParams &params) noexcept = 0;
};