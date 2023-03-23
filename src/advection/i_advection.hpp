#pragma once

#include "AdvectionParams.h"
#include <sycl/sycl.hpp>

class IAdvectorX {
public:
    virtual ~IAdvectorX() = default;

    virtual void operator()(sycl::queue &Q,
                            sycl::buffer<double, 2> &buff_fdistrib,
                            ADVParams m_params) const = 0;
};