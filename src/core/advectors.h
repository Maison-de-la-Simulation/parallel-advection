#pragma once
#include "IAdvectorX.h"

/* Contains headers for different implementations of advector interface */
namespace AdvX {

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    sycl::buffer<double, 3> m_global_buff_ftmp;

  public:
    BasicRange(const size_t nx, const size_t nvx, const size_t nz)
        : m_global_buff_ftmp{sycl::range<3>(nvx, nx, nz)} {}

    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// class BasicRange2D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange2D(const size_t nx, const size_t nvx, const size_t nz)
//         : BasicRange(nx, nvx, nz){};
// };

// class BasicRange1D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange1D(const size_t nx, const size_t nvx, const size_t nz)
//         : BasicRange(nx, nvx, nz){};
// };

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// class FakeAdvector : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;

//     sycl::event stream_bench(sycl::queue &Q,
//                              sycl::buffer<double, 1> &buff);
// };

// class HierarchicalAlloca : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;
// };

// /* Fixed memory footprint using a basic range */
// class FixedMemoryFootprint : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;
// };

// =============================================================================
// EXPERIMENTS
// =============================================================================
class MemSafe : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

class ReducedPrecision : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

}   // namespace AdvX
