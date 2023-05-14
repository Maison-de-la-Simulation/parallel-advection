#pragma once
#include "IAdvectorX.h"

/* Contains headers for different implementations of advector interface */
namespace AdvX {

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    // std::unique_ptr<sycl::buffer<double, 2>> m_global_buff_ftmp;
    mutable sycl::buffer<double, 2> m_global_buff_ftmp;

  public:
    BasicRange(const size_t nx, const size_t nvx)
        : m_global_buff_ftmp{sycl::range<2>(nvx, nx)} {}
};

class BasicRange2D : public BasicRange {
  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;

    explicit BasicRange2D(const size_t nx, const size_t nvx)
        : BasicRange(nx, nvx){};
};

class BasicRange1D : public BasicRange {
  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;

    explicit BasicRange1D(const size_t nx, const size_t nvx)
        : BasicRange(nx, nvx){};
};

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class HierarchicalAlloca : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

/* Fixed memory footprint using a basic range */
class FixedMemoryFootprint : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

}   // namespace AdvX