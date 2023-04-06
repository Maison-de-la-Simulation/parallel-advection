#pragma once
#include "i_advector_x.h"

class BasicRangeAdvector : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};