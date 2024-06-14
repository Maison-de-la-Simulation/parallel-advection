#include "advectors.h"

HIPSYCL_UNIVERSAL_TARGET
void optimized_codepaths(double* ptr, int ib, int nx, int ns, int idx_ipos1, int is, double& value)
{
    // Only executed on CUDA device. CUDA specific device functions can be called here
  __hipsycl_if_target_cuda(
        value = __ldg(ptr + (ib*nx*ns+idx_ipos1*ns+is));
    // [&](){
        // value = __ldg(ptr + (ib*nx*ns+idx_ipos1*ns+is));
    // }
  );
}

sycl::event
AdvX::Exp1::operator()(sycl::queue &Q,
                               sycl::buffer<double, 3> &buff_fdistrib,
                               const ADVParams &params) {
    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{nb, 1, ns};
    const sycl::range wg_sise{1, params.wg_size_x, 1};
    // const sycl::range wg_sise{1, nx, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_sise, [=](sycl::group<3> g) {
            // g.parallel_for_work_item(sycl::range{1, nx, 1},
            //                 [&](sycl::h_item<3> it) {
            //                     const int ix = it.get_local_id(1);
            //                     const int ib = g.get_group_id(0);
            //                     const int is = g.get_group_id(2);
            //                     __ldg();
            //                     fdist[ib][ix][is] = slice_ftmp[ix];
            //                 });

            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {

                    double* ptr = fdist.get_pointer();

                    const int ix = it.get_local_id(1);
                    const int ib = g.get_group_id(0);
                    const int is = g.get_group_id(2);


                    double const xFootCoord = displ(ix, ib, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;
                        // Only executed on CUDA device. CUDA specific device functions can be called here
                    slice_ftmp[ix] = 0.;

                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        double v = 0;
                        optimized_codepaths(ptr, ib, nx, ns, idx_ipos1, is, v);
                        // auto value = __ldg(ptr + (ib*nx*ns+idx_ipos1*ns+is));
                        // double value = __ldg(&fdist[ivx][idx_ipos1][iz]);

                        // auto value = *(ptr + (ib*nx*ns+idx_ipos1*ns+is));

                        // slice_ftmp[ix] += coef[k] * fdist[ib][idx_ipos1][iz];
                        slice_ftmp[ix] += coef[k] * v;
                    }
                });   // end parallel_for_work_item --> Implicit barrier
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
            g.parallel_for_work_item(sycl::range{1, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int ib = g.get_group_id(0);
                                         const int is = g.get_group_id(2);

                                         fdist[ib][ix][is] = slice_ftmp[ix];
                                     });
#else
            g.async_work_group_copy(fdist.get_pointer()
                                        + g.get_group_id(2)
                                        + g.get_group_id(0) *ns*nx, /* dest */
                                    slice_ftmp.get_pointer(), /* source */
                                    nx, /* n elems */
                                    ns  /* stride */
            );
#endif
        });   // end parallel_for_work_group
    });       // end Q.submit
}
