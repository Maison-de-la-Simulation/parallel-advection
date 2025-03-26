#include "init.h"
#include "unique_ref.h"
#include <advectors.h>

// To switch case on a str
[[nodiscard]] constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be: {BasicRange3D, NDRange, AdaptiveWg}";

// // ==========================================
// // ==========================================
sref::unique_ref<IAdvectorX>
kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params,
                    AdvectionSolver &s) {
    std::string kernel_name(params.kernelImpl.begin(), params.kernelImpl.end());

    switch (str2int(kernel_name.data())) {
    case str2int("BasicRange"):
        return sref::make_unique<AdvX::BasicRange>(params.n1, params.n0,
                                                   params.n2);
    case str2int("NDRange"):
        return sref::make_unique<AdvX::NDRange>();
    // case str2int("Hierarchical"):
    //     return sref::make_unique<AdvX::Hierarchical>();
    case str2int("AdaptiveWg"):
        return sref::make_unique<AdvX::AdaptiveWg>(params, q);
    // case str2int("HybridMem"):
    //     return sref::make_unique<AdvX::HybridMem>(params, q);
    default:
        auto str = kernel_name + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
    }
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, double* fdist_dev,
            const ADVParams &params) {
    const auto n0=params.n0, n1=params.n1, n2=params.n2;

    sycl::range r3d(n0,n1,n2);
    q.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(r3d, [=](auto i) {
             mdspan3d_t fdist(fdist_dev, n0, n1, n2);
             const size_t i0 = i[0];
             const size_t i1 = i[1];
             const size_t i2 = i[2];

             double x = params.minRealX + i1 * params.dx;
             fdist(i0, i1, i2) = sycl::sin(4 * x * M_PI);
         });      // end parallel_for
     }).wait();   // end q.submit
}
