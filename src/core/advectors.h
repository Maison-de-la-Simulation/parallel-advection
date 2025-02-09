#pragma once
#include "IAdvectorX.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <experimental/mdspan>

using real_t = double;

using mdspan3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;
using mdspan2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

/* Contains headers for different implementations of advector interface */
namespace AdvX {
using buff3d = sycl::buffer<double, 3>;

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    buff3d m_global_buff_ftmp;

  public:
    BasicRange(const size_t n1, const size_t nvx, const size_t n2)
        : m_global_buff_ftmp{sycl::range<3>(nvx, n1, n2)} {}

    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// =============================================================================
class AdaptiveWg : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

    AdaptiveWgDispatch wg_dispatch_;
    BlockingConfig1D bconf_d0_;
    BlockingConfig1D bconf_d2_;

    /* We should be able to query max_batchs to the API. 
              |    x    |   y/z   |
        CUDA:| 2**31-1 | 2**16-1 |
        HIP :| 2**32-1 | 2**32-1 |
        L0  :| 2**32-1 | 2**32-1 | (compile with -fno-sycl-query-fit-in-int)
        CPU : a lot
    */
    // const size_t max_batchs_x_ = 2147483648-1;
    // const size_t max_batchs_yz_ = 65536-1;
    const size_t max_batchs_x_ = 100;
    const size_t max_batchs_yz_ = 100;

    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver,
                                   const sycl::range<3> &global_size,
                                   const sycl::range<3> &local_size,
                                   const BlockingDispatch1D &block_n0,
                                   const BlockingDispatch1D &block_n2);

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    AdaptiveWg() = delete;


    AdaptiveWg(const Solver &solver, sycl::queue q){
        const auto n0 = solver.params.n0;
        const auto n1 = solver.params.n1;
        const auto n2 = solver.params.n2;

        bconf_d0_ = init_1d_blocking(n0, max_batchs_x_);
        bconf_d2_ = init_1d_blocking(n2, max_batchs_yz_);

        //SYCL query returns the size in bytes
        auto max_elem_local_mem =
            q.get_device().get_info<sycl::info::device::local_mem_size>() /
            sizeof(double);

        auto ideal_wg_dispatch = compute_ideal_wg_size(
            solver.params.pref_wg_size, max_elem_local_mem, n0, n1, n2);

        // Precompute adaptive work-group sizes
        wg_dispatch_ = compute_adaptive_wg_dispatch(ideal_wg_dispatch,
                                                    bconf_d0_, bconf_d2_);

        std::cout << "--------------------------------"    << std::endl;
        std::cout << "n_batch0       : " << bconf_d0_.n_batch_ << std::endl;
        std::cout << "last_n0_offset : " << bconf_d0_.last_dispatch_.offset_ << std::endl;
        std::cout << "last_batch_size_0 : " << bconf_d0_.last_dispatch_.batch_size_ << std::endl;
        
        std::cout << std::endl;
        
        std::cout << "n_batch2       : " << bconf_d2_.n_batch_ << std::endl;
        std::cout << "last_n2_offset : " << bconf_d2_.last_dispatch_.offset_ << std::endl;
        std::cout << "last_batch_size_2 : " << bconf_d2_.last_dispatch_.batch_size_ << std::endl;

        std::cout << std::endl;

        std::cout << "max_elems_alloc: " << max_elem_local_mem << std::endl;
        std::cout << "--------------------------------"    << std::endl;
    
    }
};

// =============================================================================
class HybridMem : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    WgDispatch wg_dispatch_;
    BlockingDispatch1D batchs_dispatch_d0_;
    BlockingDispatch1D batchs_dispatch_d2_;

    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver,
                                   const BlockingDispatch1D &block_n0,
                                   const BlockingDispatch1D &block_n2,
                                   const KernelDispatch &k_dispatch);
};

// // =============================================================================
// class HybridMem : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;
//     sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
//                                  const Solver &solver,
//                                  const size_t &ny_batch_size,
//                                  const size_t &ny_offset, const size_t k_global,
//                                  const size_t k_local);

//     void init_batchs(const Solver &s) {
//         /* Compute number of batchs */
//         float div =
//             static_cast<float>(s.p.n0) / static_cast<float>(MAX_N0_BATCHS_);
//         auto floor_div = std::floor(div);
//         auto div_is_int = div == floor_div;
//         n_batch_ = div_is_int ? div : floor_div + 1;

//         last_n0_size_ = div_is_int ? MAX_N0_BATCHS_ : (s.p.n0 % MAX_N0_BATCHS_);
//         last_n0_offset_ = MAX_N0_BATCHS_ * (n_batch_ - 1);
//     }

//     /* Initiate how many local/global kernels will be running*/
//     void init_splitting(const Solver &solver) {
//         auto div = solver.p.n0 < MAX_N0_BATCHS_
//                        ? solver.p.n0 * solver.p.percent_loc
//                        : MAX_N0_BATCHS_ * solver.p.percent_loc;
//         k_local_ = std::floor(div);

//         k_global_ = solver.p.n0 < MAX_N0_BATCHS_ ? solver.p.n0 - k_local_
//                                                  : MAX_N0_BATCHS_ - k_local_;

//         if (n_batch_ > 1) {
//             last_k_local_ = std::floor(last_n0_size_ * solver.p.percent_loc);
//             last_k_global_ = last_n0_size_ - last_k_local_;
//         } else {
//             last_k_local_ = k_local_;
//             last_k_global_ = k_global_;
//         }
//     }

//     /* Max number of batch submitted */
//     static constexpr size_t MAX_N0_BATCHS_ = 65535;
//     // static constexpr size_t PREF_WG_SIZE_ = 128;   // for A100
//     // static constexpr size_t MAX_LOCAL_ALLOC_ = 6144;

//     size_t max_elem_local_mem_;

//     size_t n_batch_;
//     size_t last_n0_size_;
//     size_t last_n0_offset_;

//     /* Number of kernels to run in global memory */
//     // float p_local_kernels = 0.5; //half by default
//     size_t k_local_;
//     size_t k_global_;
//     size_t last_k_global_;
//     size_t last_k_local_;

//     size_t loc_wg_size_0_ = 1;   // TODO: set this as in alg latex
//     size_t loc_wg_size_1_;
//     size_t loc_wg_size_2_;

//     size_t glob_wg_size_0_ = 1;   // TODO: set this as in alg latex
//     size_t glob_wg_size_1_;
//     size_t glob_wg_size_2_;

//     sycl::queue q_;
//     double *scratchG_;

//   public:
//     sycl::event operator()(sycl::queue &Q, double *fdist_dev,
//                            const Solver &solver) override;

//     HybridMem() = delete;

//     // TODO: gérer le cas ou percent_loc est 1 ou 0 (on fait tou dans la local
//     // mem ou tout dnas la global)
//     HybridMem(const Solver &solver, const sycl::queue &q) : q_(q) {
//         init_batchs(solver);
//         init_splitting(solver);

//         //SYCL query returns the size in bytes
//         max_elem_local_mem_ =
//             q.get_device().get_info<sycl::info::device::local_mem_size>() /
//             sizeof(double); //TODO: this can be templated by ElemType

//         auto n1 = solver.p.n1;
//         auto n2 = solver.p.n2;
//         auto pref_wg_size = solver.p.pref_wg_size;

//         /* Global kernels wg sizes */
//         glob_wg_size_0_ = 1;
//         if (n2 >= pref_wg_size) {
//             glob_wg_size_1_ = 1;
//             glob_wg_size_2_ = pref_wg_size;
//         } else {
//             if (n1 * n2 >= pref_wg_size) {
//                 glob_wg_size_1_ = pref_wg_size / n2;
//                 glob_wg_size_2_ = n2;
//             } else {
//                 // Not enough n1*n2 to fill up work group, we use more n0
//                 glob_wg_size_0_ = std::floor(pref_wg_size / n1 * n2);
//                 glob_wg_size_1_ = n1;
//                 glob_wg_size_2_ = n2;
//             }
//         }

//         if (glob_wg_size_2_ * n1 >= max_elem_local_mem_) {
//             loc_wg_size_2_ = std::floor(max_elem_local_mem_ / n1);
//             loc_wg_size_1_ = std::floor(pref_wg_size / loc_wg_size_2_);
//             loc_wg_size_0_ = 1;
//         } else {
//             loc_wg_size_0_ = glob_wg_size_0_;
//             loc_wg_size_1_ = glob_wg_size_1_;
//             loc_wg_size_2_ = glob_wg_size_2_;
//         }

//         if (k_global_ > 0) {
//             scratchG_ = sycl::malloc_device<double>(
//                 k_global_ * solver.p.n1 * solver.p.n2, q);
//             std::cout << "Allocated " << k_global_ << "*" << solver.p.n1 << "*"
//                       << solver.p.n2 << "bytes in memory for scartchG"
//                       << std::endl;
//         } else {
//             scratchG_ = nullptr;
//         }


//         std::cout << "--------------------------------"    << std::endl;
//         std::cout << "n_batch        : " << n_batch_        << std::endl;
//         std::cout << "k_local        : " << k_local_        << std::endl;
//         std::cout << "k_global       : " << k_global_       << std::endl;
//         std::cout << "last_k_global  : " << last_k_global_  << std::endl;
//         std::cout << "last_k_local   : " << last_k_local_   << std::endl;
//         std::cout << "last_n0_offset : " << last_n0_offset_ << std::endl;
//         std::cout << "max_elems_alloc: " << max_elem_local_mem_ << std::endl;
//         std::cout << "--------------------------------"    << std::endl;
//     }

//     ~HybridMem() {
//         if (scratchG_ != nullptr)
//             sycl::free(scratchG_, q_);
//     }
// };

}   // namespace AdvX
