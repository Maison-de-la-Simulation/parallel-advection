#include <Adv4dParams.hpp>
#include <VxSolver.hpp>
#include <iostream>
#include <sycl/sycl.hpp>
#include <init.hpp>
#include <validation.hpp>

#include <bkma.hpp>
#include <types.hpp>
// #include <impl_selector.hpp>

BkmaOptimParams create_params_adv4d(sycl::queue &q, const Adv4dParams &params){
    const auto& n0 = params.n0;
    const auto& n1 = params.n1;
    const auto& n2 = params.n2;

    WorkItemDispatch wi_dispatch;
    wi_dispatch.set_ideal_sizes(params.pref_wg_size, n0, n1, n2);
    auto max_elem_local_mem =
        q.get_device().get_info<sycl::info::device::local_mem_size>() /
        sizeof(real_t);
    wi_dispatch.adjust_sizes_mem_limit(max_elem_local_mem, n1);

    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, 1, 1,
                                    wi_dispatch.w0_, wi_dispatch.w2_);

    constexpr auto M0 = 65535;//262144;//4294967296-1;//65535;
    constexpr auto M2 = 65535;//4294967296-1;
    // constexpr auto M0 = M2;

    BatchConfig1D bconf_d0 = init_1d_blocking(n0, M0);
    BatchConfig1D bconf_d2 = init_1d_blocking(n2, M2);

    /* TODO : here compute the number of batchs */
    return BkmaOptimParams{
        bconf_d0,         // BatchConfig1D dispatch_d0
        bconf_d2,         // BatchConfig1D dispatch_d2
        wi_dispatch.w0_,     // size_t w0
        wi_dispatch.w1_,     // size_t w1
        wi_dispatch.w2_,     // size_t w2
        wg_dispatch,         // WorkGroupDispatch wg_disp
        MemorySpace::Local};
}

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "4d-advection.ini";
    ConfigMap configMap(input_file);

    Adv4dParams params;
    params.setup(configMap);
    params.print();

    sycl::queue Q{};
    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";


    params.n0 = params.nx * params.ny;
    params.n1 = params.nvx;
    params.n2 = params.nvy;

    const auto &n0=params.n0, n1=params.n1, n2=params.n2;

    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    span2d_t efield(sycl_alloc(params.nx*params.ny, Q), params.nx, params.ny);
    Q.wait();

    std::cout << "Filling" << std::endl;
    // fill_buffer_4d_adv(Q, data, params);
    
    std::cout << "Creating params" << std::endl;
    // AdvectionSolver solver(params);
    VxSolver solverVx(params, efield);

    auto optim_params = create_params_adv4d(Q, params);
    
    std::cout << "Selecting impl" << std::endl;
    auto bkma_vx = bkma_run<VxSolver, BkmaImpl::AdaptiveWg>;
    
    std::cout << "Time loop" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    bkma_vx(Q, data, solverVx, optim_params, span3d_t{});
    Q.wait();
    
    //n0 = ...
    // bkma_run_function(Q, data, solverVy, optim_params, span3d_t{});
    // Q.wait();

    //Transpose data

    // bkma_run_function(Q, data, solverX, optim_params, span3d_t{});
    // Q.wait();

    //n0 = ...
    // bkma_run_function(Q, data, solverY, optim_params, span3d_t{});
    // Q.wait();
        
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;
    
    // std::cout << "Validating" << std::endl;
    // validate_result_adv(Q, data, params);
    
    std::cout << "End" << std::endl;
    auto const n_cells = n0 * n1 * n2;
    print_perf(elapsed_seconds.count(), n_cells);

    sycl::free(data.data_handle(), Q);
    Q.wait();
    return 0;
}
