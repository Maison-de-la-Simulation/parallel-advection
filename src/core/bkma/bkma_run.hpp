#pragma once
#include <bkma_tools.hpp>
#include <MemorySpace.hpp>
#include <types.hpp>

#include <BasicRange.hpp>
#include <NDRange.hpp>
#include <AdaptiveWg.hpp>

template <class MySolver, BkmaImpl Impl>
inline sycl::event
bkma_run(sycl::queue &Q, span3d_t data, const MySolver &solver,
         BkmaOptimParams optim_params, span3d_t global_scratch = span3d_t{}) {

    sycl::event last_event;

    auto const &n_batch0 = optim_params.dispatch_d0.n_batch_;
    auto const &n_batch2 = optim_params.dispatch_d2.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = optim_params.dispatch_d0.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = optim_params.dispatch_d2.offset(i2_batch);

            auto &batch_size_d0 =
                last_i0 ? optim_params.dispatch_d0.last_batch_size_
                        : optim_params.dispatch_d0.batch_size_;
            auto &batch_size_d2 =
                last_i2 ? optim_params.dispatch_d2.last_batch_size_
                        : optim_params.dispatch_d2.batch_size_;

            last_event.wait();
            switch (optim_params.mem_space) {
            case MemorySpace::Local: {
                last_event = submit_kernels<MemorySpace::Local, MySolver, Impl>(
                    Q, data, solver, batch_size_d0, offset_d0,
                    batch_size_d2, offset_d2, optim_params.w0, optim_params.w1,
                    optim_params.w2, optim_params.wg_dispatch);
            } break;

            case MemorySpace::Global: {
                last_event =
                    submit_kernels<MemorySpace::Global, MySolver, Impl>(
                        Q, data, solver, batch_size_d0, offset_d0,
                        batch_size_d2, offset_d2, optim_params.w0,
                        optim_params.w1, optim_params.w2,
                        optim_params.wg_dispatch, global_scratch);
            } break;

            default: {
                throw std::invalid_argument("Unknown MemorySpace");
            }
            } // end switch

        } // end for i2_batch
    } // end for i0_batch

    return last_event;
} // end bkma_run
