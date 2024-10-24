#pragma once

#include "AdvectionParams.h"
#include <sycl/sycl.hpp>
#include <Solver.h>

class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const ADVParams &params) = 0;

};
