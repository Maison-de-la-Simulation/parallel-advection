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
    "Scoped, HierarchicalAlloca, HierarchicalMallocDevice}";

// // ==========================================
// // ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
kernel_impl_factory(const ADVParams &params) {
    std::string kernel_name = params.kernelImpl.data();
    switch (str2int(kernel_name.data())) {
    case str2int("Sequential"):
        return sref::make_unique<AdvX::Sequential>();
        break;
    case str2int("BasicRange2D"):
        return sref::make_unique<AdvX::BasicRange2D>(params.nx, params.nVx);
        break;
    case str2int("BasicRange1D"):
        return sref::make_unique<AdvX::BasicRange1D>(params.nx, params.nVx);
        break;
    case str2int("Hierarchical"):
        return sref::make_unique<AdvX::Hierarchical>();
        break;
    case str2int("HierarchicalAlloca"):
        return sref::make_unique<AdvX::HierarchicalAlloca>();
        break;
    case str2int("HierarchicalMallocDevice"):
        return sref::make_unique<AdvX::HierarchicalMallocDevice>();
        break;
    case str2int("NDRange"):
        return sref::make_unique<AdvX::NDRange>();
        break;
    case str2int("Scoped"):
        return sref::make_unique<AdvX::Scoped>();
        break;
    default:
        auto str = kernel_name + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
        break;
    }
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &buff_fdist,
            const ADVParams &params) noexcept {

    sycl::host_accessor fdist(buff_fdist, sycl::write_only, sycl::no_init);

    for (int ix = 0; ix < params.nx; ++ix) {
        for (int iv = 0; iv < params.nVx; ++iv) {
            double x = params.minRealx + ix * params.dx;
            fdist[iv][ix] = sycl::sin(4 * x * M_PI);
        }
    }
}