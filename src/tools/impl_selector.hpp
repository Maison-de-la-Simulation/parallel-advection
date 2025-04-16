#pragma once
#include <algorithm>
#include <string>
#include <cctype>
#include <types.hpp>
#include <sycl/sycl.hpp>
#include <bkma.hpp>

static constexpr auto error_str =
    "Should be: {BasicRange, NDRange, AdaptiveWg}";

// ==========================================
// ==========================================
[[nodiscard]] inline constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

// ==========================================
// ==========================================
[[nodiscard]] std::string
to_lowercase(const std::string &input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// ==========================================
// ==========================================
template <typename Solver>
std::function<
    sycl::event(sycl::queue &, span3d_t, const Solver &, BkmaOptimParams,
                span3d_t)> inline impl_selector(const std::string &impl_name) {

    auto impl = to_lowercase(impl_name);
    switch (str2int(impl.data())) {
    // case str2int("basicrange"):
    //     return &bkma_run<Solver, BkmaImpl::BasicRange>;
    // case str2int("ndrange"):
    //     return &bkma_run<Solver, BkmaImpl::NDRange>;
    case str2int("adaptivewg"):
        return &bkma_run<Solver, BkmaImpl::AdaptiveWg>;
    default:
        auto str =
            impl_name + " is not a valid implementation name.\n" + error_str;
        throw std::runtime_error(str);
    }
}

// bkma_run<ConvSolver, BkmaImpl::AdaptiveWg>(Q, warmup_data, solver,
//     optim_params)

// sref::unique_ref<IAdvectorX>
// kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params,
//                     AdvectionSolver &s) {
//     std::string kernel_name(params.kernelImpl.begin(),
//     params.kernelImpl.end());

//     switch (str2int(kernel_name.data())) {
//     case str2int("BasicRange"):
//         return sref::make_unique<AdvX::BasicRange>(s, q);
//     case str2int("NDRange"):
//         return sref::make_unique<AdvX::NDRange>();
//     case str2int("AdaptiveWg"):
//         return sref::make_unique<AdvX::AdaptiveWg>(s, q);
//     default:
//         auto str = kernel_name + " is not a valid kernel name.\n" +
//         error_str; throw std::runtime_error(str);
//     }
// }

