#pragma once
#include <AdvectionParams.hpp>
#include <sycl/sycl.hpp>
#include "types.hpp"

// ==========================================
// ==========================================
real_t validate_result(sycl::queue &Q, real_t* fdist_dev,
                       const ADVParams &params, bool do_print=true);
