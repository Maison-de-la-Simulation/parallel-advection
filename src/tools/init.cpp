#include "init.h"
#include "unique_ref.h"
#include <advectors.h>

// To switch case on a str
[[nodiscard]] constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be: {Sequential, BasicRange, "
    "Hierarchical, NDRange, "
    "Scoped, StreamY, StraddledMalloc, Exp1...}";

// // ==========================================
// // ==========================================
sref::unique_ref<IAdvectorX>
kernel_impl_factory(const sycl::queue &q, const ADVParamsNonCopyable &params) {
    std::string kernel_name(params.kernelImpl.begin(), params.kernelImpl.end());

    switch (str2int(kernel_name.data())) {
    // case str2int("Sequential"):
    //     return sref::make_unique<AdvX::Sequential>();
    case str2int("BasicRange"):
        return sref::make_unique<AdvX::BasicRange>(params.n1, params.n0,
                                                   params.n2);
    // case str2int("BasicRange1D"):
    // return sref::make_unique<AdvX::BasicRange1D>(params.n1, params.n0);
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
    // case str2int("StreamY"):
    //     return sref::make_unique<AdvX::StreamY>();
    // case str2int("StraddledMalloc"):
    //     return sref::make_unique<AdvX::StraddledMalloc>();
    // case str2int("ReducedPrecision"):
    //     return sref::make_unique<AdvX::ReducedPrecision>();
    // case str2int("ReverseIndexes"):
    //     return sref::make_unique<AdvX::ReverseIndexes>();
    // case str2int("TwoDimWG"):
    //     return sref::make_unique<AdvX::TwoDimWG>();
    // case str2int("SeqTwoDimWG"):
    //     return sref::make_unique<AdvX::SeqTwoDimWG>();
    // case str2int("Exp1"):
    //     return sref::make_unique<AdvX::Exp1>(params, q);
    case str2int("Exp2"):
        return sref::make_unique<AdvX::Exp2>(params, params.percent_loc, q);
    case str2int("Exp3"):
        return sref::make_unique<AdvX::Exp3>(params, q);
    case str2int("Exp4"):
        return sref::make_unique<AdvX::Exp4>(params);
    case str2int("Exp5"):
        return sref::make_unique<AdvX::Exp5>(params);
    case str2int("Exp6"):
        return sref::make_unique<AdvX::Exp6>(params, q);
    case str2int("Alg5"):
        return sref::make_unique<AdvX::Exp6>(params, q);
    // case str2int("CudaLDG"):
    //     return sref::make_unique<AdvX::CudaLDG>();
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

    sycl::range r3d(params.n0, params.n1, params.n2);
    q.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
             mdspan3d_t fdist(fdist_dev, r3d.get(0), r3d.get(1), r3d.get(2));
             const int i1 = itm[1];

             double x = params.minRealX + i1 * params.dx;
             fdist(itm[0], itm[1], itm[2]) = sycl::sin(4 * x * M_PI);
         });      // end parallel_for
     }).wait();   // end q.submit
}
