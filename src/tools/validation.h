#pragma once
#include <AdvectionParams.h>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
double validate_result(sycl::queue &Q, double* fdist_dev,
                       const ADVParams &params, bool do_print=true);
