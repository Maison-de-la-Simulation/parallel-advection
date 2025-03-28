#pragma once
#include "unique_ref.h"
#include <sycl/sycl.hpp>
#include <advectors.h>
#include "types.h"

// ==========================================
// ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params,
                    AdvectionSolver &s);

// ==========================================
// ==========================================
void fill_buffer(sycl::queue &q, real_t* fidst_dev,
                        const ADVParams &params);
