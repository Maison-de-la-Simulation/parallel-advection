#pragma once
#include "IAdvectorX.hpp"
#include "bkma.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <experimental/mdspan>

/* Contains headers for different implementations of advector interface */
namespace AdvX {

//==============================================================================


//==============================================================================
class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
                           const AdvectionSolver &solver) override;
};


}   // namespace AdvX
