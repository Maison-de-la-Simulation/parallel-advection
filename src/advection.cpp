#include <AdvectionParams.hpp>
#include <AdvectionSolver.hpp>
#include <iostream>
#include <sycl/sycl.hpp>
#include <unique_ref.hpp>
#include <init.hpp>
#include <validation.hpp>

#include <bkma.hpp>
#include <types.hpp>
#include <impl_selector.hpp>

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);

    ADVParamsNonCopyable strParams;// = ADVParamsNonCopyable();
    strParams.setup(configMap);

    const bool run_on_gpu = strParams.gpu;
    auto device = pick_device(run_on_gpu);
    strParams.gpu = device.is_gpu() ? true : false;

    sycl::queue Q{device};

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
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    Q.wait();
    fill_buffer_adv(Q, data, params);
    
    AdvectionSolver solver(params);
    auto optim_params = create_optim_params<ADVParams>(Q, params);
    auto bkma_run_function = impl_selector<AdvectionSolver>(strParams.kernelImpl);

    // ==== Setup subgroups params =====
    const auto N_Subgroups = params.nSubgroups_Local + params.nSubgroups_Global;
    const auto Total_SeqSize =
        (params.nSubgroups_Local * params.seqSize_Local +
         params.nSubgroups_Global * params.seqSize_Global) /
        N_Subgroups;
    const auto N_groups_in_dim0 = (n0 / Total_SeqSize) / N_Subgroups;

    // ==== global scratch alloc ==== (contiguous in n1)
    span3d_t global_scratch(
        sycl_alloc(params.nSubgroups_Global*N_groups_in_dim0 * n1 * n2, Q),
        params.nSubgroups_Global*N_groups_in_dim0, n2, n1);
    Q.wait();

    auto start = std::chrono::high_resolution_clock::now();
    // Time loop
    for (size_t t = 0; t < maxIter; ++t) {
        bkma_run_function(Q, data, solver, optim_params, global_scratch);
        Q.wait();

    }   // end for t < T
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    validate_result_adv(Q, data, params);

    auto const n_cells = n0 * n1 * n2 * (maxIter);
    print_perf(elapsed_seconds.count(), n_cells);

    sycl::free(global_scratch.data_handle(), Q);
    sycl::free(data.data_handle(), Q);
    Q.wait();
    return 0;
}
