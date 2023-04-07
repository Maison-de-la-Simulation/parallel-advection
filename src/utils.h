#include <AdvectionParams.h>
#include <advectors.h>
#include <iostream>
#include <sycl/sycl.hpp>

// To switch case on a str
constexpr unsigned int
str2int(const char *str, int h = 0) {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str = "Should be one of: {Sequential, BasicRange, "
                                  "BasicRange1D, Hierarchical, NDRange, "
                                  "Scoped, MultiDevice}";

// // ==========================================
// // ==========================================
std::unique_ptr<IAdvectorX>
getKernelImpl(std::string k) {
    switch (str2int(k.data())) {
    case str2int("Sequential"):
        return std::make_unique<AdvX::Sequential>();
        break;
    case str2int("BasicRange"):
        return std::make_unique<AdvX::BasicRange>();
        break;
    case str2int("BasicRange1D"):
        return std::make_unique<AdvX::BasicRange1D>();
        break;
    case str2int("Hierarchical"):
        return std::make_unique<AdvX::Hierarchical>();
        break;
    case str2int("NDRange"):
        return std::make_unique<AdvX::NDRange>();
        break;
    case str2int("Scoped"):
        return std::make_unique<AdvX::Scoped>();
        break;
    case str2int("MultiDevice"):
        return std::make_unique<AdvX::MultiDevice>();
        break;
    }
    auto str = k + " is not a valid kernel name.\n" + error_str;
    throw std::runtime_error(str);
    return nullptr;
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &fdist,
            const ADVParams &params) {
    q.submit([&](sycl::handler &cgh) {
         sycl::accessor FDIST(fdist, cgh, sycl::write_only, sycl::no_init);

         cgh.parallel_for(fdist.get_range(), [=](sycl::id<2> itm) {
             double x = itm[1];
             for (int ivx = 0; ivx < params.nVx; ++ivx) {
                 FDIST[ivx][itm[1]] =
                     ivx % 2 == 0 ? sycl::sin(x) : sycl::cos(x);
             }
         });
     }).wait();   // end q.submit
}

// ==========================================
// ==========================================
void
print_buffer(sycl::buffer<double, 2> &fdist, const ADVParams &params) {
    sycl::host_accessor tab(fdist, sycl::read_only);

    for (int iv = 0; iv < params.nVx; ++iv) {
        for (int ix = 0; ix < params.nx; ++ix) {
            std::cout << tab[iv][ix] << " ";
        }
        std::cout << std::endl;
    }
}   // end print_buffer

// ==========================================
// ==========================================
double
check_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
             const ADVParams &params, const bool _DEBUG) {
    /* Fill a buffer the same way we filled fdist at init */
    sycl::buffer<double, 2> buff_init(sycl::range<2>(params.nVx, params.nx));
    fill_buffer(Q, buff_init, params);

    if (_DEBUG) {
        std::cout << "\nFdist_init :" << std::endl;
        print_buffer(buff_init, params);
    }

    /* Check norm of difference, should be 0 */
    sycl::buffer<double, 2> buff_res(buff_init.get_range());
    Q.submit([&](sycl::handler &cgh) {
         auto A = buff_init.get_access<sycl::access::mode::read>(cgh);
         auto B = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
         sycl::accessor C(buff_res, cgh, sycl::write_only, sycl::no_init);

         cgh.parallel_for(buff_init.get_range(), [=](auto itm) {
             C[itm] = A[itm] - B[itm];
             C[itm] *= C[itm];   // We square each elements
         });
     }).wait_and_throw();

    if (_DEBUG) {
        std::cout << "\nDifference Buffer :" << std::endl;
        print_buffer(buff_res, params);
    }

    double sumResult = 0;
    {
        sycl::buffer<double> buff_sum{&sumResult, 1};

        Q.submit([&](sycl::handler &cgh) {
             // Input values to reductions are standard accessors
             auto inputValues =
                 buff_res.get_access<sycl::access_mode::read>(cgh);

#ifdef __INTEL_LLVM_COMPILER   // for DPCPP
             auto sumReduction = sycl::reduction(buff_sum, cgh, sycl::plus<>());
#else   // for openSYCL
             auto sumAcc = buff_sum.get_access<sycl::access_mode::write>(cgh);
             auto sumReduction = sycl::reduction(sumAcc, sycl::plus<double>());
#endif
             cgh.parallel_for(buff_res.get_range(), sumReduction,
                              [=](auto idx, auto &sum) {
                                  // plus<>() corresponds to += operator, so sum
                                  // can be updated via += or combine()
                                  sum += inputValues[idx];
                              });
         }).wait_and_throw();
    }

    return std::sqrt(sumResult);
}   // end check_result
