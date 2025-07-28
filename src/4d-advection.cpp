#include <AdvectionParams.hpp>
#include <AdvectionSolver.hpp>
#include <iostream>
#include <sycl/sycl.hpp>
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
    // std::string input_file = argc > 1 ? std::string(argv[1]) : "advection.ini";
    // ConfigMap configMap(input_file);

    // ADVParamsNonCopyable strParams;// = ADVParamsNonCopyable();
    // strParams.setup(configMap);

    // const bool run_on_gpu = strParams.gpu;
    // auto device = pick_device(run_on_gpu);
    // strParams.gpu = device.is_gpu() ? true : false;

    // sycl::queue Q{device};

    /* Display infos on current device */
    // std::cout << "Using device: "
            //   << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    /* Make trivially copyable params based on strParams*/
    // strParams.print();
    // ADVParams params(strParams);

    auto n0 = params.x * params.y;
    auto n1 = params.vx;
    auto n2 = params.vy;

    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    Q.wait();

    std::cout << "Filling" << std::endl;
    fill_buffer_4d_adv(Q, data, params);
    
    std::cout << "Creating params" << std::endl;
    // AdvectionSolver solver(params);
    VxSolver solverVx(params);

    // auto optim_params = create_optim_params<ADVParams>(Q, params);
    
    std::cout << "Selecting impl" << std::endl;
    auto bkma_run_function = impl_selector<AdvectionSolver>(strParams.kernelImpl);
    
    std::cout << "Time loop" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    bkma_run_function(Q, data, solverVx, optim_params, span3d_t{});
    Q.wait();
    
    //n0 = 
    bkma_run_function(Q, data, solverVy, optim_params, span3d_t{});
    Q.wait();

    //Transpose data

    bkma_run_function(Q, data, solverX, optim_params, span3d_t{});
    Q.wait();
    bkma_run_function(Q, data, solverY, optim_params, span3d_t{});
    Q.wait();
        
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;
    
    std::cout << "Validating" << std::endl;
    // validate_result_adv(Q, data, params);
    
    std::cout << "End" << std::endl;
    auto const n_cells = n0 * n1 * n2 * (maxIter);
    print_perf(elapsed_seconds.count(), n_cells);

    sycl::free(data.data_handle(), Q);
    Q.wait();
    return 0;
}
