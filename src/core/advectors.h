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
    BasicRange(const size_t nx, const size_t nvx, const size_t ny1)
        : m_global_buff_ftmp{sycl::range<3>(nvx, nx, ny1)} {}

    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// class BasicRange2D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange2D(const size_t nx, const size_t nvx, const size_t
//     ny1)
//         : BasicRange(nx, nvx, ny1){};
// };

// class BasicRange1D : public BasicRange {
//   public:
//     sycl::event operator()(sycl::queue &Q,
//                            sycl::buffer<double, 3> &buff_fdistrib,
//                            const ADVParams &params) override;

//     explicit BasicRange1D(const size_t nx, const size_t nvx, const size_t
//     ny1)
//         : BasicRange(nx, nvx, ny1){};
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
class StreamY : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event actual_advection(sycl::queue &Q,
                                 sycl::buffer<double, 3> &buff_fdistrib,
                                 const ADVParams &params, const size_t &n_nvx,
                                 const size_t &ny_offset);

  public:
    // StreamY(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReducedPrecision : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class StraddledMalloc : public IAdvectorX {
    using IAdvectorX::IAdvectorX;
    sycl::event adv_opt3(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                         const ADVParams &params,
                         const size_t &nx_rest_to_malloc);

  public:
    // StraddledMalloc(const ADVParams &params);

    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class ReverseIndexes : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class TwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class SeqTwoDimWG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

// =============================================================================
class Exp1 : public IAdvectorX {

    static constexpr size_t MAX_NX_ALLOC = 64;
        // 6144;   // TODO: setup this value depending on hw
    static constexpr size_t MAX_NY_BATCH = 512;
        // 65535;   // TODO: setup this value depending on hw

    sycl::queue q_;
    size_t overslice_nx_size_; //what's left to malloc in global memory
    size_t n_batch_;
    size_t last_batch_size_ny_, last_batch_offset_ny_;

    double* buffer_rest_nx;

    void init(sycl::queue &q, const size_t nx, const size_t ny) noexcept {
        /* Get the number of batchs required */
        double div =
            static_cast<double>(ny) / static_cast<double>(MAX_NY_BATCH);
        auto floor_div = std::floor(div);
        auto div_is_int = div == floor_div;

        n_batch_ = div_is_int ? div : floor_div + 1;

        last_batch_size_ny_ =
            div_is_int && ny > MAX_NY_BATCH ? MAX_NY_BATCH : (ny % MAX_NY_BATCH);
        last_batch_offset_ny_ = MAX_NY_BATCH * (n_batch_ - 1);

        /* Get the size of the rest to malloc inside global mem */
        auto overslice_nx_size_ = nx <= MAX_NX_ALLOC ? 0 : nx - MAX_NX_ALLOC;
        
        if(overslice_nx_size_ > 0){
          buffer_rest_nx = sycl::malloc_device<double>(overslice_nx_size_*ny, q);
        }
        else{
          //TODO: what else?
        }

    }
    //TODO: constructor with percentage_malloc
    //TODO: here the straddled alloc is vertical, implement horizontal straddled

    using IAdvectorX::IAdvectorX;
    sycl::event
    actual_advection(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                     const ADVParams &params, const size_t &ny_batch_size,
                     const size_t &ny_offset);

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;

    explicit Exp1(sycl::queue &q, const ADVParams &p) : q_(q) { init(q_, p.nx, p.ny); }

    // ~Exp1() { sycl::free(buffer_rest_nx, q_); }

    // void reset_buffer(const size_t ny){
    //     for (size_t i = 0; i < ny; i++) {
    //         for (size_t j = 0; j < overslice_nx_size_; j++) {
    //           buffer_rest_nx[i*overslice_nx_size_ + j] = 0;
    //         }
    //     }
    // }
};

// =============================================================================
class CudaLDG : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q,
                           sycl::buffer<double, 3> &buff_fdistrib,
                           const ADVParams &params) override;
};

}   // namespace AdvX
