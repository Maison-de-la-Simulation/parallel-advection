#pragma once

#include <IAdvector.h>
#include "AdvectionParams.h"
#include <sycl/sycl.hpp>

class IAdvectorVx : public IAdvector {
  public:
    virtual sycl::event operator()(sycl::queue &Q,
                                   sycl::buffer<double, 3> &buff_fdistrib,
                                   sycl::buffer<double, 1> &buff_electric_field,
                                   const ADVParams &params) noexcept = 0;

};