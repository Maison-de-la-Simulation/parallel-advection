#include <AdvectionParams.h>
#include <advectors.h>
#include <sycl/sycl.hpp>
#include <iostream>

#include "tools/init.h"
#include "tools/validation.h"

// ==========================================
// ==========================================
// returns duration for maxIter-1 iterations
std::chrono::duration<double>
advection(sycl::queue &Q, double* fidst_dev,
          sref::unique_ref<IAdvectorX> &advector, const Solver &solver) {

    auto static const maxIter = solver.p.maxIter;

    auto start = std::chrono::high_resolution_clock::now();
    // Time loop
    for (size_t t = 0; t < maxIter; ++t) {
        advector(Q, fidst_dev, solver).wait();
    }   // end for t < T
    Q.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();

    return (end - start);
}   // end advection

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);

    ADVParamsNonCopyable strParams;// = ADVParamsNonCopyable();
    strParams.setup(configMap);

    const auto run_on_gpu = strParams.gpu;

    sycl::device d;
    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception e) {
            std::cout
                << "GPU was requested but none is available, running kernels "
                   "on the CPU\n"
                << std::endl;
            d = sycl::device{sycl::cpu_selector_v};
            strParams.gpu = false;
        }
    else
        d = sycl::device{sycl::cpu_selector_v};

    sycl::queue Q{d};

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    /* Make trivially copyable params based on strParams*/
    strParams.print();
    ADVParams params(strParams);

    const auto n1 = params.n1;
    const auto n0 = params.n0;
    const auto n2 = params.n2;
    const auto maxIter = params.maxIter;
    
    /* Buffer for the distribution function containing the probabilities of
    having a particle at a particular speed and position, plus a fictive dim */
    // sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(n0, n1, n2));
    double* fdist = sycl::malloc_device<double>(n0*n1*n2, Q);
    fill_buffer(Q, fdist, params);
    
    Solver solver(params);
    auto advector = kernel_impl_factory(Q, strParams, solver);

    auto elapsed_seconds = advection(Q, fdist, advector, solver);

    std::cout << "\nRESULTS_VALIDATION:" << std::endl;
    validate_result(Q, fdist, params);

    // if(params.outputSolution){
    //     export_result_to_file(buff_fdistrib, params);
    //     export_error_to_file(buff_fdistrib, params);
    // }

    std::cout << "PERF_DIAGS:" << std::endl;
    std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    auto gcells = ((n0*n1*n2*(maxIter)) / elapsed_seconds.count()) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(double) * 2
              << " GB/s" << std::endl;
    return 0;
}
