#include "IAdvectorX.h"
#include "advectors.h"

// AdvX::LargeMalloc::LargeMalloc(const ADVParams &params){
//     auto n_batch = std::ceil(params.nvx / MAX_NVX); //should be in
//     constructor

// }

// ==========================================
// ==========================================
sycl::event
AdvX::LargeMalloc::actual_advection(sycl::queue &Q,
                                sycl::buffer<double, 3> &buff_fdistrib,
                                const ADVParams &params) {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const nz = params.nz;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{nvx, 1, nz};
    const sycl::range wg_size{1, params.wg_size, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int ivx = g.get_group_id(0);
                    const int iz = g.get_group_id(2);

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

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
            g.async_work_group_copy(fdist.get_pointer()
                                        + g.get_group_id(2)
                                        + g.get_group_id(0) *nz*nx, /* dest */
                                    slice_ftmp.get_pointer(), /* source */
                                    nx, /* n elems */
                                    nz  /* stride */
            );
#endif
        });   // end parallel_for_work_group
    });       // end Q.submit
} // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::LargeMalloc::operator()(sycl::queue &Q,
                            sycl::buffer<double, 3> &buff_fdistrib,
                            const ADVParams &params) {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const nz = params.nz;

    //On A100 it breaks if we allocate more than 48 KiB per block, which is 6144 double
    //On MI250x it breaks if we allocate more than 64KiB per wg, which is 8192 double
    constexpr size_t MAX_NX_ALLOC = 6144; //A100
    if (nx < MAX_NX_ALLOC) {
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        // cudaDeviceSynchronize(); works ?
        return actual_advection(
            Q, buff_fdistrib, params);
    }
}
