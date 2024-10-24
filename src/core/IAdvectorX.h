#pragma once

#include <sycl/sycl.hpp>
#include <Solver.h>

class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver) = 0;

};
