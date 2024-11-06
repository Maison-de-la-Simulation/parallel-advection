#pragma once
#include "AdvectionParams.h"
#include "IAdvectorX.h"
#include <cmath>
#include <cstddef>
#include <experimental/mdspan>
#include <stdexcept>

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

// class BasicRange2D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            double* fdist_dev,
//                            const Solver &solver) override;

//     explicit BasicRange2D(const size_t n1, const size_t nvx, const size_t
//     n2)
//         : BasicRange(n1, nvx, n2){};
// };

// class BasicRange1D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            double* fdist_dev,
//                            const Solver &solver) override;

//     explicit BasicRange1D(const size_t n1, const size_t nvx, const size_t
//     n2)
//         : BasicRange(n1, nvx, n2){};
// };

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

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// class FakeAdvector : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            double* fdist_dev,
//                            const Solver &solver) override;

//     sycl::event stream_bench(sycl::queue &Q,
//                              sycl::buffer<double, 1> &buff);
// };

// class HierarchicalAlloca : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            double* fdist_dev,
//                            const Solver &solver) override;
// };

// /* Fixed memory footprint using a basic range */
// class FixedMemoryFootprint : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            double* fdist_dev,
//                            const Solver &solver) override;
// };

// =============================================================================
// EXPERIMENTS
// =============================================================================
class StreamY : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver, const size_t &n_nvx,
                                 const size_t &ny_offset);

  public:
    // StreamY(const Solver &solver);

    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// =============================================================================
class ReducedPrecision : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// =============================================================================
class StraddledMalloc : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event adv_opt3(sycl::queue &Q, double *fdist_dev,
                         const Solver &solver, const size_t &nx_rest_to_malloc);

  public:
    // StraddledMalloc(const Solver &solver);

    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// =============================================================================
// class ReverseIndexes : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q, double* fdist_dev,
//                            const Solver &solver) override;
// };

// =============================================================================
// class TwoDimWG : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q, double* fdist_dev,
//                            const Solver &solver) override;
// };

// =============================================================================
// class SeqTwoDimWG : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q, double* fdist_dev,
//                            const Solver &solver) override;
// };

// =============================================================================
class Exp1 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS_ = 65535;
    /* Max number of elements in the local accessor */
    // static constexpr size_t MAX_NX_ALLOC = 64;

    sycl::queue q_;
    double *global_vertical_buffer_;
    size_t n_batch_;
    size_t last_n0_size_;
    size_t last_n0_offset_;
    size_t n0_rest_malloc_;
    size_t local_alloc_size_;

    void init_batchs(const Solver &s) {
        /* Compute number of batchs */
        float div =
            static_cast<float>(s.p.n0) / static_cast<float>(MAX_NY_BATCHS_);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_n0_size_ = div_is_int ? MAX_NY_BATCHS_ : (s.p.n0 % MAX_NY_BATCHS_);
        last_n0_offset_ = MAX_NY_BATCHS_ * (n_batch_ - 1);
    }

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp1() = delete;

    Exp1(const Solver &s, const sycl::queue &q) : q_(q) {
        init_batchs(s);

        local_alloc_size_ = std::floor(s.p.percent_loc * s.p.n0);
        n0_rest_malloc_ = s.p.n1 - local_alloc_size_;

        if (n0_rest_malloc_ > 0) {
            // TODO: don't allocate full n0, only the current batch_size_ny size
            global_vertical_buffer_ = sycl::malloc_device<double>(
                s.p.n0 * n0_rest_malloc_ * s.p.n2, q_);
        } else {
            global_vertical_buffer_ = nullptr;
        }
    }

    ~Exp1() {
        if (n0_rest_malloc_ > 0)
            sycl::free(global_vertical_buffer_, q_);
    }
};

// =============================================================================
class Exp2 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    void init_batchs(const Solver &s) {
        /* Compute number of batchs */
        float div =
            static_cast<float>(s.p.n0) / static_cast<float>(MAX_NY_BATCHS_);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_n0_size_ = div_is_int ? MAX_NY_BATCHS_ : (s.p.n0 % MAX_NY_BATCHS_);
        last_n0_offset_ = MAX_NY_BATCHS_ * (n_batch_ - 1);
    }

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS_ = 65535;

    size_t n_batch_;
    size_t last_n0_size_;
    size_t last_n0_offset_;

    /* Number of kernels to run in global memory */
    // float p_local_kernels = 0.5; //half by default
    size_t k_local_;
    size_t k_global_;

    sycl::queue q_;
    double *global_buffer_;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp2() = delete;

    Exp2(const Solver &solver) {
        init_batchs(solver);
        k_global_ = 0;
        k_local_ = solver.p.n0 * solver.p.n2;
    }

    // TODO: gérer le cas ou percent_loc est 1 ou 0 (on fait tou dans la local
    // mem ou tout dnas la global)
    Exp2(const Solver &solver, const float percent_in_local_mem_per_ny1_slice,
         const sycl::queue &q)
        : q_(q) {
        init_batchs(solver);

        /* n_kernel_per_ny1 = solver.p.n0; TODO: attention ça c'est vrai
         seulement quand n0 < MAX_NY et qu'on a un seul batch, sinon on le
         percentage doit s'appliquer pour chaque taille de batch_ny!!! */
        auto div = solver.p.n0 * percent_in_local_mem_per_ny1_slice;
        k_local_ = std::floor(div);
        k_global_ = solver.p.n0 - k_local_;

        if (k_global_ > 0) {
            global_buffer_ = sycl::malloc_device<double>(
                k_global_ * solver.p.n1 * solver.p.n2, q);
        } else {
            global_buffer_ = nullptr;
        }
    }

    ~Exp2() {
        if (global_buffer_ != nullptr)
            sycl::free(global_buffer_, q_);
    }
};

// =============================================================================
class Exp3 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    sycl::queue q_;
    size_t n_batch_;
    size_t last_n0_size_;
    size_t last_n0_offset_;

    size_t concurrent_ny_slices_;

    double *scratch_;

    void init_batchs(const Solver &s) {
        /* Compute number of batchs */
        double div = static_cast<float>(s.p.n0) /
                     static_cast<float>(concurrent_ny_slices_);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_n0_size_ = div_is_int ? concurrent_ny_slices_
                                   : (s.p.n0 % concurrent_ny_slices_);
        last_n0_offset_ = concurrent_ny_slices_ * (n_batch_ - 1);
    }

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp3() = delete;

    Exp3(const Solver &solver, const sycl::queue &q) : q_(q) {
        /*TODO: this percent_loc is not actually percent_in_local_memory but is
        used to obtain CONCURRENT_NY_SLICES, we could specify this value with a
        MAX_GLOBAL_MEM_ALLOC size or directly with the number of concurrent n0
        slices we want to process or a max size to not exceed */
        auto div = solver.p.n0 * solver.p.percent_loc;
        concurrent_ny_slices_ = std::floor(div);
        /*TODO: check concurrent_ny_slices_ does not exceed max memory available
        when creating the buffer*/

        init_batchs(solver);

        scratch_ = sycl::malloc_device<double>(
            concurrent_ny_slices_ * solver.p.n1 * solver.p.n2, q_);
    }

    ~Exp3() { sycl::free(scratch_, q_); }
};

// =============================================================================
class Exp4 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    static constexpr size_t MAX_ALLOC_SIZE_ = 6144;   // TODO this is for A100

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp4() = delete;

    Exp4(const Solver &solver) {
        if (solver.p.n1 * solver.p.n2 > MAX_ALLOC_SIZE_) {
            throw std::runtime_error(
                "n1*n0 > MAX_ALLOC_SIZE_: a single slice of the problem cannot "
                "fit in local memory, Exp4 not possible");
        }
    }
};

// =============================================================================
class Exp5 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    static constexpr size_t MAX_ALLOC_SIZE_ = 6144;   // TODO this is for A100
    static constexpr size_t WARP_SIZE_ = 32;          // for A100
    static constexpr size_t PREF_WG_SIZE_ = 128;      // for A100

    size_t wg_size_1_;
    size_t wg_size_2_;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp5() = delete;

    Exp5(const Solver &solver) {

        wg_size_1_ = std::ceil(PREF_WG_SIZE_ / solver.p.n2);
        wg_size_2_ = wg_size_1_ > 1 ? solver.p.n2 : PREF_WG_SIZE_;

        /*
        3 cas:
          - n2 == 32 : we do 1nx at the time
          - n2 > 32: we do 1nx at the time (we finish n1 slice before starting
          other n2 to limit memory footprint)
          - n2 < 32 : we do floor(warp_size/n2) n1 at the time

         we can adjust the wg_size with the max size of local accessor, but
         in the best case we prefer to use PREF_WG_SIZE as total number of
        threads
        */
    }
};

// =============================================================================
class Exp6 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);
    /*Same as Exp5 but in global memory*/

    static constexpr size_t MAX_ALLOC_SIZE_ = 6144;   // TODO this is for A100
    static constexpr size_t WARP_SIZE_ = 32;          // for A100
    static constexpr size_t PREF_WG_SIZE_ = 128;      // for A100

    size_t wg_size_1_;
    size_t wg_size_2_;
    sycl::queue q_;
    double *scratch_;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Exp6() = delete;

    Exp6(const Solver &solver, const sycl::queue &q) : q_(q) {

        wg_size_1_ = std::ceil(PREF_WG_SIZE_ / solver.p.n2);
        wg_size_2_ = wg_size_1_ > 1 ? solver.p.n2 : PREF_WG_SIZE_;

        /* TODO: allocate only for concurrent slice in dim0*/
        scratch_ = sycl::malloc_device<double>(
            solver.p.n0 * solver.p.n1 * solver.p.n2, q_);
    }

    ~Exp6() { sycl::free(scratch_, q_); }
};

// =============================================================================
class Alg5 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
                                 const Solver &solver,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset, const size_t k_global,
                                 const size_t k_local);

    void init_batchs(const Solver &s) {
        /* Compute number of batchs */
        float div =
            static_cast<float>(s.p.n0) / static_cast<float>(MAX_NY_BATCHS_);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_n0_size_ = div_is_int ? MAX_NY_BATCHS_ : (s.p.n0 % MAX_NY_BATCHS_);
        last_n0_offset_ = MAX_NY_BATCHS_ * (n_batch_ - 1);
    }

    /* Initiate how many local/global kernels will be running*/
    void init_splitting(const Solver &solver) {
        auto div = solver.p.n0 < MAX_NY_BATCHS_
                       ? solver.p.n0 * solver.p.percent_loc
                       : MAX_NY_BATCHS_ * solver.p.percent_loc;
        k_local_ = std::floor(div);

        k_global_ = solver.p.n0 < MAX_NY_BATCHS_ ? solver.p.n0 - k_local_
                                                 : MAX_NY_BATCHS_ - k_local_;

        if (n_batch_ > 1) {
            last_k_local_  = std::floor(last_n0_size_ * solver.p.percent_loc);
            last_k_global_ = last_n0_size_ -last_k_local_;
        } else {
            last_k_local_  = k_local_;
            last_k_global_ = k_global_;
        }
    }

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS_ = 65535;
    static constexpr size_t PREF_WG_SIZE_ = 128;   // for A100

    size_t n_batch_;
    size_t last_n0_size_;
    size_t last_n0_offset_;

    /* Number of kernels to run in global memory */
    // float p_local_kernels = 0.5; //half by default
    size_t k_local_;
    size_t k_global_;
    size_t last_k_global_;
    size_t last_k_local_;

    size_t wg_size_0_ = 1; //TODO: set this as in alg latex
    size_t wg_size_1_;
    size_t wg_size_2_;

    sycl::queue q_;
    double *scratchG_;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    Alg5() = delete;

    // Alg5(const Solver &solver) {
    //     init_batchs(solver);
    //     k_global_ = 0;
    //     k_local_ = solver.p.n0 * solver.p.n2;
    // }

    // TODO: gérer le cas ou percent_loc est 1 ou 0 (on fait tou dans la local
    // mem ou tout dnas la global)
    Alg5(const Solver &solver, const sycl::queue &q) : q_(q) {
        init_batchs(solver);
        init_splitting(solver);

        wg_size_1_ = std::ceil(PREF_WG_SIZE_ / solver.p.n2);
        wg_size_2_ = wg_size_1_ > 1 ? solver.p.n2 : PREF_WG_SIZE_;

        /* TODO: allocate only for concurrent slice in dim0*/
        scratchG_ = sycl::malloc_device<double>(
            solver.p.n0 * solver.p.n1 * solver.p.n2, q_);

        if (k_global_ > 0) {
            scratchG_ = sycl::malloc_device<double>(
                k_global_ * solver.p.n1 * solver.p.n2, q);
        } else {
            scratchG_ = nullptr;
        }
    }

    ~Alg5() {
        if (scratchG_ != nullptr)
            sycl::free(scratchG_, q_);
    }
};

// =============================================================================
// class CudaLDG : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q, double* fdist_dev,
//                            const Solver &solver) override;
// };

}   // namespace AdvX
