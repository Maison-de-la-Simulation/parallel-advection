#pragma once
#include <IAdvectorVx.h>

namespace advector {

namespace vx {

class Hierarchical : public IAdvectorVx {
    using IAdvectorVx::IAdvectorVx;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 2> &buff_fdistrib,
                           sycl::buffer<double, 1> &buff_electric_field,
                           const ADVParams &params) noexcept override;
};

}   // namespace vx

} // namespace advector