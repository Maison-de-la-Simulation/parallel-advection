#pragma once
#include <IAdvectorX.h>

/* Contains headers for different implementations of advector interface */
namespace advector {

namespace x {

class MDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    void operator()(KV_double_3d &fdistrib,
                    const ADVParams &params) noexcept override;
};

}   // namespace x

}   // namespace advector
