#include "kokkos_shortcut.hpp"
#include <AdvectionParams.h>
#include <InitParams.h>
#include <init.h>
#include <validation.h>
#include <io.h>
#include "unique_ref.h"

// ==========================================
// ==========================================
void
advection(KV_double_3d &fdistrib,
          KV_double_1d &efield,
          sref::unique_ref<IAdvectorX> &x_advector,
          sref::unique_ref<IAdvectorVx> &vx_advector,
          const ADVParams &runParams) {

    auto static const maxIter = runParams.maxIter;
    
    // Time loop, cannot parallelize this
    for (auto t = 0; t < maxIter; ++t) {
        x_advector(fdistrib, runParams);
        // vx_advector(fdistrib, efield, runParams);
    }   // end for t < T

}   // end advection


// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    Kokkos::ScopeGuard const kguard;

    /* Display infos */
    Kokkos::print_configuration(std::cout);

    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);
    ADVParams runParams = ADVParams();
    runParams.setup(configMap);
    runParams.print();

    InitParams initParams = InitParams();
    initParams.setup(configMap);
    initParams.print_kokkos();

    const auto nx = runParams.nx;
    const auto nvx = runParams.nvx;
    const auto n_fict_dim = runParams.n_fict_dim;
    const auto maxIter = runParams.maxIter;

    KV_double_3d fdist("fdist", n_fict_dim, nvx, nx);


    /* Fictive electric field to advect along vx */
    // std::vector<double> elec{nx, 0};
    KV_double_1d efield("efield", nx);

    fill_buffers(fdist, efield, runParams);

    auto x_advector = x_advector_factory(runParams, initParams);
    auto vx_advector = vx_advector_factory(runParams);

    auto start = std::chrono::high_resolution_clock::now();
    advection(fdist, efield, x_advector, vx_advector, runParams);
    // advection(fdist, efield, x_advector, runParams);
    auto end = std::chrono::high_resolution_clock::now();

    Kokkos::fence("main_scope_fence"); //not sure about that

    std::cout << "\nRESULTS_VALIDATION:" << std::endl;
    validate_result(fdist, runParams, initParams);

    if (initParams.outputSolution) {
        export_result_to_file(fdist, runParams);
        export_error_to_file(fdist, runParams);
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