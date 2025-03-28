#pragma once
#include "unique_ref.hpp"
#include <sycl/sycl.hpp>
#include <advectors.hpp>
#include "types.hpp"

// ==========================================
// ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params,
                    AdvectionSolver &s);

// ==========================================
// ==========================================
void fill_buffer(sycl::queue &q, real_t* fidst_dev,
                        const ADVParams &params);
