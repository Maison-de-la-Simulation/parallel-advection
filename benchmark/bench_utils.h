#pragma once

#include <AdvectionParams.h>
#include <sycl/sycl.hpp>
#include <init.h>
#include <advectors.h>

enum AdvImpl : int {
    BR2D,   // 0
    BR1D,   // 1
    HIER,   // 2
    NDRA,   // 3
    SCOP    // 4
};

// =============================================
// =============================================
[[nodiscard]] inline sycl::queue
createSyclQueue(const bool run_on_gpu, benchmark::State &state) {
    sycl::device d;

    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::runtime_error e) {
            state.SkipWithError("GPU was requested but none is available, skipping benchmark.");
        }
    else
        d = sycl::device{sycl::cpu_selector_v};
    return sycl::queue{d};
}   // end createSyclQueue

// =============================================
// =============================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
advectorFactory(const AdvImpl kernel_id, const size_t nx, const size_t nvx,
                benchmark::State &state) {
    ADVParamsNonCopyable params;
    params.nx = nx;
    params.nvx = nvx;

    switch (kernel_id) {
    case AdvImpl::BR2D:
        params.kernelImpl = "BasicRange2D";
        break;
    case AdvImpl::BR1D:
        params.kernelImpl = "BasicRange1D";
        break;
    case AdvImpl::HIER:
        params.kernelImpl = "Hierarchical";
        break;
    case AdvImpl::NDRA:
        params.kernelImpl = "NDRange";
        break;
    case AdvImpl::SCOP:
        params.kernelImpl = "Scoped";
        break;

    default:
        auto str = "Error: wrong kernel_id.\n";
        throw std::runtime_error(str);
        break;
    }

    return kernel_impl_factory(params);
}   // end advectorFactory
