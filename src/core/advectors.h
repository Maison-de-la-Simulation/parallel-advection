#pragma once
#include "AdvectionParams.h"
#include "IAdvectorX.h"

/* Contains headers for different implementations of advector interface */
namespace AdvX {
using buff3d = sycl::buffer<double, 3>;

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
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

    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// class BasicRange2D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange2D(const size_t nx, const size_t nvx, const size_t ny1)
//         : BasicRange(nx, nvx, ny1){};
// };

// class BasicRange1D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            buff3d &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange1D(const size_t nx, const size_t nvx, const size_t ny1)
//         : BasicRange(nx, nvx, ny1){};
// };

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
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
    sycl::event actual_advection(sycl::queue &Q,
                                 buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &n_nvx,
                                 const size_t &ny_offset);

  public:
    // StreamY(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReducedPrecision : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class StraddledMalloc : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event adv_opt3(sycl::queue &Q,
                         buff3d &buff_fdistrib,
                         const ADVParams &params,
                        const size_t &nx_rest_to_malloc);

  public:
    // StraddledMalloc(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReverseIndexes : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class TwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class SeqTwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class Exp1 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q,
                                 buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);


    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS   = 128;
    /* Max number of elements in the local accessor */
    static constexpr size_t MAX_NX_ALLOC    = 64;

    sycl::queue q_;
    double* global_vertical_buffer_;
    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;
    size_t nx_rest_malloc_;

    void init_batchs(const ADVParams &p){
        /* Compute number of batchs */
        double div =
            static_cast<double>(p.ny) / static_cast<double>(MAX_NY_BATCHS);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ = div_is_int ? MAX_NY_BATCHS : (p.ny % MAX_NY_BATCHS);
        last_ny_offset_ = MAX_NY_BATCHS * (n_batch_ - 1);
    }

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;


    Exp1() = delete;

    Exp1(const ADVParams &p, const sycl::queue &q) : q_(q) {
      init_batchs(p);


      nx_rest_malloc_ = p.nx <= MAX_NX_ALLOC ? 0 : p.nx - MAX_NX_ALLOC;

      if(nx_rest_malloc_ > 0){
        //TODO: don't allocate full ny, only the actual batch_size_ny
        global_vertical_buffer_ = sycl::malloc_device<double>(p.ny * nx_rest_malloc_, q_);
      }
      else {
        global_vertical_buffer_ = nullptr;
      }
    }

    ~Exp1() { sycl::free(global_vertical_buffer_, q_); }
};

// =============================================================================
class Exp2 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q,
                                 buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    void init_batchs(const ADVParams &p){
        /* Compute number of batchs */
        double div =
            static_cast<double>(p.ny) / static_cast<double>(MAX_NY_BATCHS);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ = div_is_int ? MAX_NY_BATCHS : (p.ny % MAX_NY_BATCHS);
        last_ny_offset_ = MAX_NY_BATCHS * (n_batch_ - 1);
    }

    /* Max number of batch submitted */
    static constexpr size_t MAX_NY_BATCHS   = 65535;

    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;

    /* Number of kernels to run in global memory */
    // float p_local_kernels = 0.5; //half by default
    size_t k_local_;
    size_t k_global_;

    sycl::queue q_;
    double* global_buffer_;
    

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp2() = delete;

    Exp2(const ADVParams &params){
      init_batchs(params);
      k_global_  = 0;
      k_local_ = params.ny*params.ny1;
    }

    //TODO: gérer le cas ou percent_loc est 1 ou 0 (on fait tou dans la local mem ou tout dnas la global)
    Exp2(const ADVParams &params, const float percent_in_local_mem_per_ny1_slice,
         const sycl::queue &q)
        : q_(q) {
        init_batchs(params);

        /* n_kernel_per_ny1 = params.ny; TODO: attention ça c'est vrai seulement
         quand ny < MAX_NY et qu'on a un seul batch, sinon on le percentage doit
         s'appliquer pour chaque taille de batch_ny!!! */
        auto div = params.ny * percent_in_local_mem_per_ny1_slice;
        k_local_ = std::floor(div);
        k_global_ = params.ny - k_local_;

        global_buffer_ =
            sycl::malloc_device<double>(k_global_ * params.nx, q);

        // std::cout << "k_global:" << k_global_ << " k_local:" << k_local_ << std::endl;
    }

    ~Exp2(){sycl::free(global_buffer_, q_);}
};

// =============================================================================
class CudaLDG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

}   // namespace AdvX
