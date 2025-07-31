#include <Adv4dParams.hpp>
#include <Advectors4d.hpp>
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

void transpose_for_x(sycl::queue &Q, const span3d_t &input, span3d_t &output, const Adv4dParams &params) {
    const int nx  = params.nx;
    const int ny  = params.ny;
    const int nvx = params.nvx;
    const int nvy = params.nvy;

    Q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<3>(nvx * nvy, nx, ny), [=](sycl::id<3> idx) {
            int iv_flat = idx[0];     // [0, nvx*nvy)
            int x       = idx[1];     // [0, nx)
            int y       = idx[2];     // [0, ny)

            int vx = iv_flat / nvy;
            int vy = iv_flat % nvy;

            int input_n0 = (x * ny + y) * nvx + vx;
            int input_n1 = vy;
            int input_n2 = 0;

            real_t val = input(input_n0, input_n1, input_n2);

            int output_n0 = vx * nvy + vy;
            int output_n1 = x;
            int output_n2 = y;

            output(output_n0, output_n1, output_n2) = val;
        });
    }).wait();
}

inline void setup_params_vx(Adv4dParams &params){
    params.n0 = params.nx * params.ny;
    params.n1 = params.nvx;
    params.n2 = params.nvy;
}

inline void setup_params_vy(Adv4dParams &params){
    params.n0 = params.nx * params.ny * params.nvx;
    params.n1 = params.nvy;
    params.n2 = 1;
}

inline void setup_params_x(Adv4dParams &params){
    params.n0 = params.nvx * params.nvy;
    params.n1 = params.nx;
    params.n2 = params.ny;
}

inline void setup_params_y(Adv4dParams &params){
    params.n0 = params.nvx * params.nvy * params.nx;
    params.n1 = params.ny;
    params.n2 = 1;
}

template <typename TimePoint>
inline void print_runtime(std::string_view name,
                          TimePoint start,
                          TimePoint end){
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << name << " ==== Kernel time: "<< elapsed_seconds.count() << " seconds\n";
}


int main(int argc, char **argv) {
    std::string input_file = argc > 1 ? std::string(argv[1]) : "4d-advection.ini";
    ConfigMap configMap(input_file);

    Adv4dParams params;
    params.setup(configMap);
    params.print();

    sycl::queue Q{};
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    setup_params_vx(params);
    auto optim_params = create_params_adv4d(Q, params);

    const auto& nx  = params.nx;
    const auto& ny  = params.ny;
    const auto& nvx = params.nvx;
    const auto& nvy = params.nvy;

    const auto N = params.n0 * params.n1 * params.n2;
    auto ptr = sycl_alloc(N, Q);
    auto ptr2 = sycl_alloc(N, Q);
    span3d_t data_vx(ptr, nx * ny      , nvx, nvy);
    span3d_t data_vy(ptr, nx * ny * nvx, nvy, 1);

    span3d_t data_x(ptr2, nvx * nvy     , nx , ny);
    span3d_t data_y(ptr2, nvx * nvy * nx, ny , 1);
    
    // auto oop_ptr = sycl_alloc(N,Q);
    // span3d_t scratch_vx(oop_ptr, nx * ny      , nvx, nvy);
    // span3d_t scratch_vy(oop_ptr, nx * ny * nvx, nvy, 1);
    // span3d_t scratch_x(oop_ptr, nvx * nvy     , nx , ny);
    // span3d_t scratch_y(oop_ptr, nvx * nvy * nx, ny , 1);
    
    // span3d_t scratch(sycl_alloc(N, Q), params.n0, params.n1, params.n2);
    span2d_t efield(sycl_alloc(params.nx * params.ny, Q), params.nx, params.ny);
    Q.wait();

    std::cout << "Filling initial data..." << std::endl;
    // fill_buffer_4d_adv(Q, data, params);


    // === Vx Advection ===
    VxSolver solverVx(params, efield);

    auto start = std::chrono::high_resolution_clock::now();
    bkma_run<VxSolver, BkmaImpl::AdaptiveWg>(Q, data_vx, solverVx, optim_params, span3d_t{}/*scratch_vx*/);
    Q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    print_runtime("GridVx", start, end);

    // === Vy Advection ===
    setup_params_vy(params);
    optim_params = create_params_adv4d(Q, params); //updating params
    VySolver solverVy(params, efield);

    start = std::chrono::high_resolution_clock::now();
    bkma_run<VySolver, BkmaImpl::AdaptiveWg>(Q, data_vy, solverVy, optim_params, span3d_t{}/*scratch_vy*/);
    Q.wait();
    end = std::chrono::high_resolution_clock::now();
    print_runtime("GridVy", start, end);

    // === Transpose for space advections ===
    std::cout << "Transposing for X/Y solvers..." << std::endl;
    transpose_for_x(Q, data_vy, data_x, params);
    

    // === X Advection ===
    setup_params_x(params);
    optim_params = create_params_adv4d(Q, params); //updating params
    XSolver solverX(params, efield);

    start = std::chrono::high_resolution_clock::now();
    bkma_run<XSolver, BkmaImpl::AdaptiveWg>(Q, data_x, solverX, optim_params, span3d_t{});
    Q.wait();
    end = std::chrono::high_resolution_clock::now();
    print_runtime("GridX", start, end);

    // === Y Advection: notranspose ===
    setup_params_y(params);
    optim_params = create_params_adv4d(Q, params); //updating params
    YSolver solverY(params, efield);

    start = std::chrono::high_resolution_clock::now();
    bkma_run<YSolver, BkmaImpl::AdaptiveWg>(Q, data_y, solverY, optim_params, span3d_t{});
    Q.wait();
    end = std::chrono::high_resolution_clock::now();
    print_runtime("GridX", start, end);

    sycl::free(efield.data_handle(), Q);
    // sycl::free(oop_ptr, Q);
    sycl::free(ptr, Q);
    sycl::free(ptr2, Q);
    Q.wait();

    return 0;
}