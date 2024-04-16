#pragma once
#include <AdvectionParams.h>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
double validate_result(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params, bool do_print=true);