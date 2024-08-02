#include "IAdvectorX.h"
#include "advectors.h"

// AdvX::StreamY::StreamY(const ADVParams &params){
//     auto n_batch = std::ceil(params.nb0 / MAX_nb); //should be in
//     constructor

// }

// ==========================================
// ==========================================
sycl::event
AdvX::StreamY::actual_advection(sycl::queue &Q,
                                sycl::buffer<double, 3> &buff_fdistrib,
                                const ADVParams &params,
                                const size_t &n_nb,
                                const size_t &nb_offset) {

    auto const nx = params.nx;
    auto const nb0 = params.nb0;
    auto const nb1 = params.nb1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{n_nb, 1, nb1};
    const sycl::range wg_size{1, params.wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int ivx = g.get_group_id(0) + nb_offset;
                    const int iz = g.get_group_id(2);

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
            g.parallel_for_work_item(sycl::range{1, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int ivx = g.get_group_id(0);
                                         const int iz = g.get_group_id(2);

                                         fdist[ivx][ix][iz] = slice_ftmp[ix];
                                     });
#else
            g.async_work_group_copy(fdist.get_pointer() + g.get_group_id(2) +
                                        (g.get_group_id(0)+nb_offset) * nb1 * nx, /* dest */
                                    slice_ftmp.get_pointer(), /* source */
                                    nx,                       /* n elems */
                                    nb1                        /* stride */
            );
#endif
        });   // end parallel_for_work_group
    });       // end Q.submit
} // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::StreamY::operator()(sycl::queue &Q,
                            sycl::buffer<double, 3> &buff_fdistrib,
                            const ADVParams &params) {
    auto const nx = params.nx;
    auto const nb0 = params.nb0;
    auto const nb1 = params.nb1;

    // IFDEF ACPP_TARGETS=cuda:sm_80 ... ?

    // On A100 it breaks when nb0 (the first dimension) is >= 65536.
    constexpr size_t MAX_nb = 65536;
    if (nb0 < MAX_nb) {
        /* If limit not exceeded we return a classical Hierarchical advector */
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        auto n_batch = std::floor(nb0 / MAX_nb)+1;  // should be in constructor

        for (int i_batch = 0; i_batch < n_batch - 1; ++i_batch) {   // can we parallel_for this on multiple GPUs? multiple queues ? or other CUDA streams?

            size_t nb_offset = (i_batch * MAX_nb) - i_batch;

            // sycl::buffer sub_buff(buff_fdistrib,
            //                       sycl::id(nb_offset, 0, 0) /*offset*/,
            //                       sycl::range(MAX_nb - 1, nx, nb1) /*range*/);

            actual_advection(Q, buff_fdistrib, params, MAX_nb-1, nb_offset).wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
        auto const nb_size = (nb0 % MAX_nb) + (n_batch - 1);
        auto const nb_offset = (MAX_nb-1)*(n_batch-1);

        return actual_advection(
            Q, buff_fdistrib, params,
            nb_size,
            nb_offset);
    }
}
