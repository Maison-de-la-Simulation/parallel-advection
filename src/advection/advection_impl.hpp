#include "i_advection.hpp"

class AdvectorX : IAdvectorX {
  public:
    void operator()(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                    ADVParams m_params) const;
};