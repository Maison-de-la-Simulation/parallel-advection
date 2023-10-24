#pragma once

#include "unique_ref.h"
#include <AdvectionParams.h>
#include <advectors.h>
#include <sycl/sycl.hpp>

// To switch case on a str
[[nodiscard]] constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be one of: {Sequential, BasicRange2D, "
    "BasicRange1D, Hierarchical, NDRange, "
    "Scoped}";//, HierarchicalAlloca, FixedMemoryFootprint}";

// // ==========================================
// // ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
kernel_impl_factory(const ADVParamsNonCopyable &params) {
    std::string kernel_name(params.kernelImpl.begin(), params.kernelImpl.end());

    switch (str2int(kernel_name.data())) {
    case str2int("Sequential"):
        return sref::make_unique<AdvX::Sequential>();
    case str2int("BasicRange2D"):
        return sref::make_unique<AdvX::BasicRange2D>(params.nx, params.nvx);
    case str2int("BasicRange1D"):
        return sref::make_unique<AdvX::BasicRange1D>(params.nx, params.nvx);
    case str2int("Hierarchical"):
        return sref::make_unique<AdvX::Hierarchical>();
    // case str2int("HierarchicalAlloca"):
    //     return sref::make_unique<AdvX::HierarchicalAlloca>();
    // case str2int("FixedMemoryFootprint"):
    //     return sref::make_unique<AdvX::FixedMemoryFootprint>();
    case str2int("NDRange"):
        return sref::make_unique<AdvX::NDRange>();
    case str2int("Scoped"):
        return sref::make_unique<AdvX::Scoped>();
    default:
        auto str = kernel_name + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
    }
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &buff_fdist,
            const ADVParams &params) noexcept {

    q.submit([&](sycl::handler &cgh) {
        sycl::accessor fdist(buff_fdist, cgh, sycl::write_only, sycl::no_init);

        cgh.parallel_for(buff_fdist.get_range(), [=](sycl::id<2> itm) {
            const int ix = itm[1];

            double x = params.minRealX + ix * params.dx;
            fdist[itm] = sycl::sin(4 * x * M_PI);
        });   // end parallel_for
    });       // end q.submit
}