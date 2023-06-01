#pragma once

#include "unique_ref.h"
#include <AdvectionParams.h>
#include <x_advectors.h>
#include <vx_advectors.h>
#include <sycl/sycl.hpp>

// To switch case on a str
[[nodiscard]] constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be one of: {Sequential, BasicRange3D, "
    "BasicRange1D, Hierarchical, NDRange, "
    "Scoped, HierarchicalAlloca, FixedMemoryFootprint}";

// // ==========================================
// // ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
x_advector_factory(const ADVParams &params) {
    std::string kernel_name = params.kernelImpl.data();
    switch (str2int(kernel_name.data())) {
    case str2int("Sequential"):
        return sref::make_unique<advector::x::Sequential>();
    case str2int("BasicRange3D"):
        return sref::make_unique<advector::x::BasicRange3D>(params.n_fict_dim,
                                                            params.nvx,
                                                            params.nx);
    case str2int("BasicRange1D"):
        return sref::make_unique<advector::x::BasicRange1D>(params.n_fict_dim,
                                                            params.nvx,
                                                            params.nx);
    case str2int("Hierarchical"):
        return sref::make_unique<advector::x::Hierarchical>();
    case str2int("HierarchicalAlloca"):
        return sref::make_unique<advector::x::HierarchicalAlloca>();
    case str2int("FixedMemoryFootprint"):
        return sref::make_unique<advector::x::FixedMemoryFootprint>();
    case str2int("NDRange"):
        return sref::make_unique<advector::x::NDRange>();
    case str2int("Scoped"):
        return sref::make_unique<advector::x::Scoped>();
    default:
        auto str = kernel_name + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
    }
}

// // ==========================================
// // ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorVx>
vx_advector_factory() {
   return sref::make_unique<advector::vx::Hierarchical>();
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 3> &buff_fdist,
            const ADVParams &params) noexcept {

    q.submit([&](sycl::handler &cgh) {
        sycl::accessor fdist(buff_fdist, cgh, sycl::write_only, sycl::no_init);

        cgh.parallel_for(buff_fdist.get_range(), [=](sycl::id<3> itm) {
            const int ix = itm[2];

            double x = params.minRealx + ix * params.dx;
            fdist[itm] = sycl::sin(4 * x * M_PI);
        });   // end parallel_for
    });       // end q.submit
}