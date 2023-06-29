#include <AdvectionParams.h>
#include <InitParams.h>
#include <x_advectors.h>
#include <vx_advectors.h>
#include <init.h>
#include <io.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include <validation.h>
#include <filesystem>

// ==========================================
// ==========================================
void
advection(sycl::queue &Q,
          sycl::buffer<double, 3> &buff_fdistrib,
          sycl::buffer<double, 1> &buff_efield,
          sref::unique_ref<IAdvectorX> &x_advector,
          sref::unique_ref<IAdvectorVx> &vx_advector,
          const ADVParams &runParams) {

    auto static const maxIter = runParams.maxIter;

    // Time loop, cannot parallelize this
    for (auto t = 0; t < maxIter; ++t) {
        x_advector(Q, buff_fdistrib, runParams);

        // If it's last iteration, we wait
        // if (t == maxIter - 1)
        //     vx_advector(Q, buff_fdistrib, buff_efield, runParams).wait_and_throw();
        // else
        //     vx_advector(Q, buff_fdistrib, buff_efield, runParams);
    }   // end for t < T

}   // end advection

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);
    ADVParams runParams = ADVParams();
    runParams.setup(configMap);

    InitParams initParams = InitParams();
    initParams.setup(configMap);

    const auto run_on_gpu = initParams.gpu;

    /* Use different queues depending on SYCL implem */
#ifdef __INTEL_LLVM_COMPILER
    std::cout << "Running with DPCPP" << std::endl;
    /* Double not supported on IntelGraphics so we choose the CPU
    if not with OpenSYCL */
    sycl::queue Q{sycl::cpu_selector_v};
#else   //__HIPSYCL__
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
            initParams.gpu = false;
        }
    else
        d = sycl::device{sycl::cpu_selector_v};

    sycl::queue Q{d};
#endif

    runParams.print();
    initParams.print();

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    const auto nx = runParams.nx;
    const auto nvx = runParams.nvx;
    const auto n_fict_dim = runParams.n_fict_dim;
    const auto maxIter = runParams.maxIter;

    /* Buffer for the distribution function containing the probabilities of
    having a particle at a particular speed and position */
    sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(n_fict_dim, nvx, nx));
    fill_buffer(Q, buff_fdistrib, runParams);

    /* Fictive electric field to advect along vx */
    std::vector<double> efield(nx, 0);
    sycl::buffer<double, 1> buff_efield(efield);

    auto x_advector = x_advector_factory(runParams, initParams);
    auto vx_advector = vx_advector_factory();

    auto start = std::chrono::high_resolution_clock::now();
    advection(Q, buff_fdistrib, buff_efield, x_advector, vx_advector,
              runParams);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "\nRESULTS_VALIDATION:" << std::endl;
    validate_result(Q, buff_fdistrib, runParams);

    if (initParams.outputSolution) {
        export_result_to_file(buff_fdistrib, runParams);
        export_error_to_file(buff_fdistrib, runParams);
    }

    std::cout << "PERF_DIAGS:" << std::endl;
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    auto gcells =
        ((nvx * nx * n_fict_dim * maxIter) / elapsed_seconds.count()) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(double) * 2
              << " GB/s" << std::endl;
    std::cout << "parsing;" << nvx * nx << ";" << nx << ";" << nvx << std::endl;
    return 0;
}
