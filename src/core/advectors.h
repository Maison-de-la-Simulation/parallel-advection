#pragma once
#include "AdvectionParams.h"
#include "IAdvectorX.h"
#include <cstddef>
#include <stdexcept>
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
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    buff3d m_global_buff_ftmp;

  public:
    BasicRange(const size_t nx, const size_t nvx, const size_t ny1)
        : m_global_buff_ftmp{sycl::range<3>(nvx, nx, ny1)} {}

    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// class BasicRange2D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange2D(const size_t nx, const size_t nvx, const size_t
//     ny1)
//         : BasicRange(nx, nvx, ny1){};
// };

// class BasicRange1D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange1D(const size_t nx, const size_t nvx, const size_t
//     ny1)
//         : BasicRange(nx, nvx, ny1){};
// };

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// class FakeAdvector : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;

//     sycl::event stream_bench(sycl::queue &Q,
//                              sycl::buffer<double, 1> &buff);
// };

// class HierarchicalAlloca : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;
// };

// /* Fixed memory footprint using a basic range */
// class FixedMemoryFootprint : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;
// };

// =============================================================================
// EXPERIMENTS
// =============================================================================
class StreamY : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                                 const ADVParams &params, const size_t &n_nvx,
                                 const size_t &ny_offset);

  public:
    // StreamY(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReducedPrecision : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class StraddledMalloc : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event adv_opt3(sycl::queue &Q, buff3d &buff_fdistrib,
                         const ADVParams &params,
                         const size_t &nx_rest_to_malloc);

  public:
    // StraddledMalloc(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReverseIndexes : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class TwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class SeqTwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class Exp1 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS = 128;
    /* Max number of elements in the local accessor */
    static constexpr size_t MAX_NX_ALLOC = 64;

    sycl::queue q_;
    double *global_vertical_buffer_;
    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;
    size_t nx_rest_malloc_;

    void init_batchs(const ADVParams &p) {
        /* Compute number of batchs */
        float div =
            static_cast<float>(p.ny) / static_cast<float>(MAX_NY_BATCHS);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ = div_is_int ? MAX_NY_BATCHS : (p.ny % MAX_NY_BATCHS);
        last_ny_offset_ = MAX_NY_BATCHS * (n_batch_ - 1);
    }

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp1() = delete;

    Exp1(const ADVParams &p, const sycl::queue &q) : q_(q) {
        init_batchs(p);

        nx_rest_malloc_ = p.nx <= MAX_NX_ALLOC ? 0 : p.nx - MAX_NX_ALLOC;

        if (nx_rest_malloc_ > 0) {
            // TODO: don't allocate full ny, only the current batch_size_ny size
            global_vertical_buffer_ =
                sycl::malloc_device<double>(p.ny * nx_rest_malloc_ * p.ny1, q_);
        } else {
            global_vertical_buffer_ = nullptr;
        }
    }

    ~Exp1() {
        if (nx_rest_malloc_ > 0)
            sycl::free(global_vertical_buffer_, q_);
    }
};

// =============================================================================
class Exp2 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    void init_batchs(const ADVParams &p) {
        /* Compute number of batchs */
        float div =
            static_cast<float>(p.ny) / static_cast<float>(MAX_NY_BATCHS);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ = div_is_int ? MAX_NY_BATCHS : (p.ny % MAX_NY_BATCHS);
        last_ny_offset_ = MAX_NY_BATCHS * (n_batch_ - 1);
    }

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS = 65535;

    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;

    /* Number of kernels to run in global memory */
    // float p_local_kernels = 0.5; //half by default
    size_t k_local_;
    size_t k_global_;

    sycl::queue q_;
    double *global_buffer_;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp2() = delete;

    Exp2(const ADVParams &params) {
        init_batchs(params);
        k_global_ = 0;
        k_local_ = params.ny * params.ny1;
    }

    // TODO: gérer le cas ou percent_loc est 1 ou 0 (on fait tou dans la local
    // mem ou tout dnas la global)
    Exp2(const ADVParams &params,
         const float percent_in_local_mem_per_ny1_slice, const sycl::queue &q)
        : q_(q) {
        init_batchs(params);

        /* n_kernel_per_ny1 = params.ny; TODO: attention ça c'est vrai seulement
         quand ny < MAX_NY et qu'on a un seul batch, sinon on le percentage doit
         s'appliquer pour chaque taille de batch_ny!!! */
        auto div = params.ny * percent_in_local_mem_per_ny1_slice;
        k_local_ = std::floor(div);
        k_global_ = params.ny - k_local_;

        if (k_global_ > 0) {
            global_buffer_ = sycl::malloc_device<double>(
                k_global_ * params.nx * params.ny1, q);
        } else {
            global_buffer_ = nullptr;
        }
    }

    ~Exp2() { if(global_buffer_ != nullptr) sycl::free(global_buffer_, q_); }
};

// =============================================================================
class Exp3 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    sycl::queue q_;
    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;

    size_t concurrent_ny_slices_;

    double* scratch_;

     void init_batchs(const ADVParams &p) {
        /* Compute number of batchs */
        double div = static_cast<float>(p.ny) /
                     static_cast<float>(concurrent_ny_slices_);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ =
            div_is_int ? concurrent_ny_slices_ : (p.ny % concurrent_ny_slices_);
        last_ny_offset_ = concurrent_ny_slices_ * (n_batch_ - 1);
    }

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp3() = delete;

    Exp3(const ADVParams &params, const sycl::queue &q) : q_(q) {
        /*TODO: this percent_loc is not actually percent_in_local_memory but is
        used to obtain CONCURRENT_NY_SLICES, we could specify this value with a
        MAX_GLOBAL_MEM_ALLOC size or directly with the number of concurrent ny
        slices we want to process or a max size to not exceed */
        auto div = params.ny * params.percent_loc;
        concurrent_ny_slices_ = std::floor(div);
        /*TODO: check concurrent_ny_slices_ does not exceed max memory available
        when creating the buffer*/

        init_batchs(params);

        scratch_ = sycl::malloc_device<double>(
            concurrent_ny_slices_ * params.nx * params.ny1, q_);
    }

    ~Exp3(){sycl::free(scratch_, q_);}

};

// =============================================================================
class Exp4 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    static constexpr size_t MAX_ALLOC_SIZE_ = 6144; //TODO this is for A100

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp4() = delete;

    Exp4(const ADVParams &params) {
        if(params.nx * params.ny1 > MAX_ALLOC_SIZE_){
            throw std::runtime_error(
                "nx*ny > MAX_ALLOC_SIZE_: a single slice of the problem cannot "
                "fit in local memory, Exp4 not possible");
        }
    }
};

// =============================================================================
class CudaLDG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

}   // namespace AdvX
