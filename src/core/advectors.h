#pragma once
#include "IAdvectorX.h"

/* Contains headers for different implementations of advector interface */
namespace AdvX
{

class Sequential : public IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    std::unique_ptr<sycl::buffer<double, 2>> m_global_buff_ftmp;

  public:
    BasicRange(const size_t &nx, const size_t &nvx){
        m_global_buff_ftmp =
            std::make_unique<sycl::buffer<double, 2>>(sycl::range<2>(nvx, nx));
    }
};

class BasicRange2D : public BasicRange {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;

    BasicRange2D(const size_t &nx, const size_t &nvx) : BasicRange(nx, nvx){};
};

class BasicRange1D : public BasicRange {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;

    BasicRange1D(const size_t &nx, const size_t &nvx) : BasicRange(nx, nvx){};
};

class Hierarchical : public IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;
};

class NDRange : public IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;
};

class Scoped : public IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const override;
};

}