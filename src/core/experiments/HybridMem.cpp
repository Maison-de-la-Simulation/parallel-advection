#include "distribute_batchs.h"
#include "submit_kernels.h"

sycl::event
AdvX::HybridMem::operator()(sycl::queue &Q, double *fdist_dev,
                                const Solver &solver) {
    return distribute_batchs(
        Q, fdist_dev, solver, dispatch_dim0_, dispatch_dim2_, local_size_.w0_,
        local_size_.w1_, local_size_.w2_, wg_dispatch_, solver.params.n0,
        solver.params.n1, solver.params.n2,
        [&](sycl::queue &Q, double *fdist_dev, const Solver &solver,
            size_t batch_size_d0, size_t offset_d0, size_t batch_size_d2,
            size_t offset_d2, size_t w0, size_t w1, size_t w2,
            WorkGroupDispatch wg_dispatch, size_t n0, size_t n1, size_t n2) {
            return submit_local_kernels(
                Q, fdist_dev, solver, kernel_dispatch_.k_local_, offset_d0,
                batch_size_d2, offset_d2, local_size_.w0_, local_size_.w1_,
                local_size_.w2_, wg_dispatch, n0, n1, n2);
        },
        [&](sycl::queue &Q, double *fdist_dev, const Solver &solver,
            size_t batch_size_d0, size_t offset_d0, size_t batch_size_d2,
            size_t offset_d2, size_t w0, size_t w1, size_t w2,
            WorkGroupDispatch wg_dispatch, size_t n0, size_t n1, size_t n2) {
            return submit_global_kernels(
                Q, fdist_dev, solver,
                kernel_dispatch_.k_global_,
                offset_d0 + kernel_dispatch_.k_local_, batch_size_d2, offset_d2,
                local_size_global_kernels_.w0_, local_size_global_kernels_.w1_,
                local_size_global_kernels_.w2_, wg_dispatch_global_kernels_, n0,
                n1, n2, global_scratch_);
        });
}


/*

class HybridMem : public AdaptiveWg {
    // Only split kernels in dim0. Too complicated for nothing to split in 2dim
    KernelDispatch kernel_dispatch_;
    KernelDispatch last_kernel_dispatch_;

    WorkItemDispatch local_size_global_kernels_;
    WorkGroupDispatch wg_dispatch_global_kernels_;

    double *global_scratch_;

    sycl::queue q_;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    HybridMem() = delete;

    HybridMem(const Solver &solver, sycl::queue q)
        : AdaptiveWg(solver, q), q_(q) {
        auto const &n0 = solver.params.n0;
        auto const &n1 = solver.params.n1;
        auto const &n2 = solver.params.n2;
        auto const &p = solver.params.percent_loc;
        // setup hybrid kernel dispatch
        kernel_dispatch_ = init_kernel_splitting(p, dispatch_dim0_.batch_size_);
        last_kernel_dispatch_ =
            init_kernel_splitting(p, dispatch_dim0_.last_batch_size_);

        // Compute local_size for global memory kernels
        local_size_global_kernels_.set_ideal_sizes(solver.params.pref_wg_size,
                                                   n0, n1, n2);
        local_size_global_kernels_.adjust_sizes_mem_limit(
            std::numeric_limits<int>::max(), n1);

        //TODO: add different seq_size parameters for k_global and k_local?
        wg_dispatch_global_kernels_.s0_ = solver.params.seq_size0;
        wg_dispatch_global_kernels_.s2_ = solver.params.seq_size2;
        //TODO: this line is overriden inside the kernel!!! useless
        wg_dispatch_global_kernels_.set_num_work_groups(
            n0, n2, dispatch_dim0_.n_batch_, dispatch_dim2_.n_batch_,
            local_size_global_kernels_.w0_, local_size_global_kernels_.w2_);

        // malloc global scratch
        auto max_k_global = std::max(kernel_dispatch_.k_global_,
                                     last_kernel_dispatch_.k_global_);
        if (max_k_global > 0) {
            global_scratch_ = sycl::malloc_device<double>(
                max_k_global * solver.params.n1 * solver.params.n2, q_);

            // std::cout << "Allocated " << max_k_global << "*" << solver.params.n1
            //           << "*" << solver.params.n2
            //           << "elems in memory for scartchG" << std::endl;
        } else {
            global_scratch_ = nullptr;
        }

        // print_range("local_size_global_k", local_size_global_kernels_.range());

        // std::cout << "(k_local, k_global) = (" << kernel_dispatch_.k_local_ << ", "
        //           << kernel_dispatch_.k_global_ << ")" << std::endl;
        // std::cout << "(last_k_local, last_k_global) = ("
        //           << last_kernel_dispatch_.k_local_ << ", "
        //           << last_kernel_dispatch_.k_global_ << ")" << std::endl;
        // std::cout << "--------------------------------" << std::endl;
    }

    ~HybridMem() {
        if (global_scratch_ != nullptr)
            sycl::free(global_scratch_, q_);
    }
};
*/