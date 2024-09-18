#pragma once
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
                                 const size_t &ny_offset,
                                 const size_t &nx_rest_to_malloc);


  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class Exp2 : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q,
                                 buff3d &buff_fdistrib,
                                 const ADVParams &params,
                                 const size_t &ny_batch_size,
                                 const size_t &ny_offset);

    static constexpr size_t MAX_NY_BATCHS   = 64;
    static constexpr float  P_LOCAL_KERNELS = 0.5;

    size_t n_batch_;
    size_t last_ny_size_;
    size_t last_ny_offset_;
    

  public:
    sycl::event operator()(sycl::queue &Q,
                           buff3d &buff_fdistrib,
                           const ADVParams &params) override;

    Exp2(const ADVParams &p){
        double div =
            static_cast<double>(p.ny) / static_cast<double>(MAX_NY_BATCHS);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;
        n_batch_ = div_is_int ? div : floor_div + 1;

        last_ny_size_ = div_is_int ? MAX_NY_BATCHS : (p.ny % MAX_NY_BATCHS);
        last_ny_offset_ = MAX_NY_BATCHS * (n_batch_ - 1);

    }
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
