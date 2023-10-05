#include <AdvectionParams.h>
#include <advectors.h>
#include <iostream>
#include <init.h>
#include <io.h>
#include <validation.h>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
void
advection(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
          sref::unique_ref<IAdvectorX> &advector, const ADVParams &params) {

    auto static const maxIter = params.maxIter;

    // Time loop, cannot parallelize this
    for (auto t = 0; t < maxIter; ++t) {

        // If it's last iteration, we wait
        if (t == maxIter - 1)
            advector(Q, buff_fdistrib, params).wait_and_throw();
        else
            advector(Q, buff_fdistrib, params);
    }   // end for t < T

}   // end advection

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);
    ADVParams params = ADVParams();
    params.setup(configMap);

    const auto run_on_gpu = params.gpu;

    /* Use different queues depending on SYCL implem */
// #if (defined(SYCL_IMPLEMENTATION_ONEAPI) || defined(__INTEL_LLVM_COMPILER))
    // std::cout << "Running with DPCPP" << std::endl;
    /* Double not supported on IntelGraphics so we choose the CPU
    if not with OpenSYCL */
    // sycl::queue Q{sycl::cpu_selector_v};
// #else   //__HIPSYCL__
    sycl::device d;
    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::runtime_error e) {
            std::cout
                << "GPU was requested but none is available, running kernels "
                   "on the CPU\n"
                << std::endl;
            d = sycl::device{sycl::cpu_selector_v};
            // d = sycl::device{sycl::};
            params.gpu = false;
        }
    else
        d = sycl::device{sycl::cpu_selector_v};

    sycl::queue Q{d};
// #endif

    params.print();

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    const auto nx = params.nx;
    const auto nvx = params.nvx;
    const auto maxIter = params.maxIter;
    
    /* Buffer for the distribution function containing the probabilities of
    having a particle at a particular speed and position */
    sycl::buffer<double, 2> buff_fdistrib(sycl::range<2>(nvx, nx));
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = kernel_impl_factory(params);

    auto start = std::chrono::high_resolution_clock::now();
    advection(Q, buff_fdistrib, advector, params);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "\nRESULTS_VALIDATION:" << std::endl;
    validate_result(Q, buff_fdistrib, params);

    if(params.outputSolution){
        export_result_to_file(buff_fdistrib, params);
        export_error_to_file(buff_fdistrib, params);
    }

    std::cout << "PERF_DIAGS:" << std::endl;
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    auto gcells = ((nvx * nx * maxIter) / elapsed_seconds.count()) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(double) * 2
              << " GB/s" << std::endl;
    std::cout << "parsing;" << nvx * nx << ";" << nx << ";" << nvx << std::endl;
    return 0;
}
