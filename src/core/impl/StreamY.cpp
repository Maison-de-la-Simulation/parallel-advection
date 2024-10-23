#include "IAdvectorX.h"
#include "advectors.h"

// AdvX::StreamY::StreamY(const ADVParams &params){
//     auto n_batch = std::ceil(params.n0 / MAX_NY); //should be in
//     constructor

// }

// ==========================================
// ==========================================
sycl::event
AdvX::StreamY::actual_advection(sycl::queue &Q, double *fdist_dev,
                                const ADVParams &params, const size_t &n_ny,
                                const size_t &ny_offset) {

    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{n_ny, 1, n2};
    const sycl::range wg_size{1, params.wg_size_1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const size_t i1 = it.get_local_id(1);
                    const size_t i0 = g.get_group_id(0) + ny_offset;
                    const size_t i2 = g.get_group_id(2);

                    double const xFootCoord = displ(i1, i0, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[i1] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int id1_ipos = (n1 + ipos1 + k) % n1;

                        slice_ftmp[i1] += coef[k] * fdist(i0, id1_ipos, i2);
                    }
                });   // end parallel_for_work_item --> Implicit barrier

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const size_t i1 = it.get_local_id(1);
                    const size_t i0 = g.get_group_id(0) + ny_offset;
                    const size_t i2 = g.get_group_id(2);

                    fdist(i0, i1, i2) = slice_ftmp[i1];
                });
#else
            g.async_work_group_copy(
                sycl::multi_ptr<double,
                                sycl::access::address_space::global_space>(
                    fdist_dev) +
                    g.get_group_id(2) +
                    (g.get_group_id(0) + ny_offset) * n2 * n1, /* dest */
                slice_ftmp.get_pointer(),                      /* source */
                n1,                                            /* n elems */
                n2                                             /* stride */
            );
#endif
        });   // end parallel_for_work_group
    });       // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::StreamY::operator()(sycl::queue &Q, double *fdist_dev,
                          const ADVParams &params) {
    // auto const n1 = params.n1;
    auto const n0 = params.n0;
    // auto const n2 = params.n2;

    // IFDEF ACPP_TARGETS=cuda:sm_80 ... ?

    // On A100 it breaks when n0 (the first dimension) is >= 65536.
    constexpr size_t MAX_NY = 128;
    if (n0 < MAX_NY) {
        /* If limit not exceeded we return a classical Hierarchical advector */
        AdvX::Hierarchical adv{};
        return adv(Q, fdist_dev, params);
    } else {
        double div = static_cast<double>(n0) / static_cast<double>(MAX_NY);
        auto floor_div = std::floor(div);
        auto is_int = div == floor_div;
        auto n_batch = is_int ? div : floor_div + 1;

        for (int i_batch = 0; i_batch < n_batch - 1;
             ++i_batch) {   // can we parallel_for this on multiple GPUs?
                            // multiple queues ? or other CUDA streams?

            size_t ny_offset = (i_batch * MAX_NY);

            actual_advection(Q, fdist_dev, params, MAX_NY, ny_offset)
                .wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
        auto const ny_size =
            is_int ? MAX_NY : (n0 % MAX_NY);   // + (n_batch - 1);
        auto const ny_offset = MAX_NY * (n_batch - 1);

        // return the last advection with the rest
        return actual_advection(Q, fdist_dev, params, ny_size, ny_offset);
    }
}
