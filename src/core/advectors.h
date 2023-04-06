#pragma once
#include "IAdvectorX.h"

/* Contains headers for different implementations of advector interface */
namespace AdvX
{

class Sequential : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class BasicRange : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class BasicRange1D : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class Hierarchical : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class NDRange : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class Scoped : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

class MultiDevice : IAdvectorX {
  public:
    sycl::event operator()(
      sycl::queue &Q,
      sycl::buffer<double, 2> &buff_fdistrib,
      const ADVParams &params) const;
};

}