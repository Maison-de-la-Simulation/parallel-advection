#pragma once
#include <IAdvectorX.h>

/* Contains headers for different implementations of advector interface */
namespace advector {

namespace x {

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    mutable sycl::buffer<double, 3> m_global_buff_ftmp;

  public:
    BasicRange(const size_t n_fict_dim, const size_t nvx, const size_t nx)
        : m_global_buff_ftmp{sycl::range<3>(n_fict_dim, nvx, nx)} {}
};

class BasicRange3D : public BasicRange {
  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;

    explicit BasicRange3D(const size_t n_fict_dim, const size_t nvx,
                          const size_t nx)
        : BasicRange(n_fict_dim, nvx, nx){};
};

class BasicRange1D : public BasicRange {
  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;

    explicit BasicRange1D(const size_t n_fict_dim, const size_t nvx,
                          const size_t nx)
        : BasicRange(n_fict_dim, nvx, nx){};
};

class Hierarchical : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class Scoped : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

class HierarchicalAlloca : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

/* Fixed memory footprint using a basic range */
class FixedMemoryFootprint : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) noexcept override;
};

}   // namespace x

}   // namespace advector
