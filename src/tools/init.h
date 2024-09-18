#pragma once
#include "unique_ref.h"
#include <sycl/sycl.hpp>
#include <advectors.h>

// ==========================================
// ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params);

// ==========================================
// ==========================================
void fill_buffer(sycl::queue &q, sycl::buffer<double, 3> &buff_fdist,
                        const ADVParams &params);
