#include "IAdvectorX.h"
#include "advectors.h"

// TODO: this is streamY, merge straddlel malloc and twoDimwg

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::actual_advection(sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &nb_batch_size,
                             const size_t &nb_offset) {

    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_b = params.wg_size_b;
    auto const wg_size_x = params.wg_size_x;

    /* nb must be divisible by slice_size_dim_y */
    if (nb % wg_size_b != 0) {
        throw std::invalid_argument("nb must be divisible by wg_size_b");
    }
    if (wg_size_b * nx > 6144) {
        /* TODO: try with a unique allocation in shared memory and sequential
         * iteration */
        throw std::invalid_argument(
            "wg_size_b*nx must be < to 6144 (shared memory limit)");
    }

    const sycl::range nb_wg{nb / wg_size_b, 1, ns};
    const sycl::range wg_size{params.wg_size_b, params.wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* We use a 2D local accessor here */
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_b, nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iz = g.get_group_id(2);

                    const int local_nb = it.get_local_id(0);
                    const int ivx = wg_size_b * g.get_group_id(0) + local_nb;

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[local_nb][ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[local_nb][ix] +=
                            coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.parallel_for_work_item(sycl::range{wg_size_b, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int iz = g.get_group_id(2);

                                         const int local_nb = it.get_local_id(0);
                                         const int ivx = wg_size_b * g.get_group_id(0) + local_nb;

                                         fdist[ivx][ix][iz] = slice_ftmp[local_nb][ix];
                                     });
        });   // end parallel_for_work_group
    });       // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::operator()(sycl::queue &Q,
                       sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {
    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;

    // On A100 it breaks when nb (the first dimension) is >= 65536.
    constexpr size_t MAX_nb = 65536;
    if (nb < MAX_nb) {
        /* If limit not exceeded we return a classical Hierarchical advector */
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        auto n_batch =
            std::floor(nb / MAX_nb) + 1;   // should be in constructor

        // For BBlock in B
        for (int i_batch = 0; i_batch < n_batch - 1; ++i_batch) {
            // can we parallel_for this on multiple GPUs?
            // multiple queues ? or other CUDA streams?

            size_t nb_offset = (i_batch * MAX_nb) - i_batch;

            actual_advection(Q, buff_fdistrib, params, MAX_nb - 1, nb_offset)
                .wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
        auto const nb_size = (nb % MAX_nb) + (n_batch - 1);
        auto const nb_offset = (MAX_nb - 1) * (n_batch - 1);

        return actual_advection(Q, buff_fdistrib, params, nb_size, nb_offset);
    }
}
