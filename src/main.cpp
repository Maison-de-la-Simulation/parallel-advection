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
advection(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
          sref::unique_ref<IAdvectorX> &advector, const ADVParams &params) {

    auto static const maxIter = params.maxIter;

    /* First iteration not timed */
    advector(Q, buff_fdistrib, params).wait_and_throw();

    auto start = std::chrono::high_resolution_clock::now();
    // Time loop
    for (auto t = 0; t < maxIter-1; ++t) {

        // If it's last iteration, we wait
        if (t == maxIter - 2)
            advector(Q, buff_fdistrib, params).wait_and_throw();
        else
            advector(Q, buff_fdistrib, params);
    }   // end for t < T
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
        } catch (const std::runtime_error e) {
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

    /* Make trivially copyable params based on strParams*/
    strParams.print();
    ADVParams params(strParams);

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    const auto nx = params.nx;
    const auto nb = params.nb;
    const auto ns = params.ns;
    const auto maxIter = params.maxIter;
    
    /* Buffer for the distribution function containing the probabilities of
    having a particle at a particular speed and position, plus a fictive dim */
    sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(nb, nx, ns));
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = kernel_impl_factory(strParams);

    auto elapsed_seconds = advection(Q, buff_fdistrib, advector, params);

    std::cout << "\nRESULTS_VALIDATION:" << std::endl;
    validate_result(Q, buff_fdistrib, params);

    // if(params.outputSolution){
    //     export_result_to_file(buff_fdistrib, params);
    //     export_error_to_file(buff_fdistrib, params);
    // }

    std::cout << "PERF_DIAGS:" << std::endl;
    std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    auto gcells = ((nb*nx*ns*(maxIter-1)) / elapsed_seconds.count()) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(double) * 2
              << " GB/s" << std::endl;
    return 0;
}
