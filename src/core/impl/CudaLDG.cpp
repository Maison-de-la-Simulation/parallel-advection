#include "advectors.h"

#ifndef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
HIPSYCL_UNIVERSAL_TARGET
void optimized_codepaths(double* ptr, int ib, int nx, int ny1, int idx_ipos1, int is, double& value)
{
  __hipsycl_if_target_cuda(
        value = __ldg(ptr + (ib*nx*ny1+idx_ipos1*ny1+is));
  );
}
#endif

sycl::event
AdvX::CudaLDG::operator()(sycl::queue &Q,
                          sycl::buffer<double, 3> &buff_fdistrib,
                          const ADVParams &params) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{ny, 1, ny1};
    const sycl::range wg_sise{1, params.wg_size_x, 1};
    // const sycl::range wg_sise{1, nx, 1};

    return Q.submit([&](sycl::handler &cgh) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
throw std::logic_error("CudaLDG kernel is not compatible with DPCPP");
#else   // for acpp
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
                        optimized_codepaths(ptr, ib, nx, ny1, idx_ipos1, is, v);
                        // auto value = __ldg(ptr + (ib*nx*ny1+idx_ipos1*ny1+is));
                        // double value = __ldg(&fdist[ivx][idx_ipos1][iz]);

                        // auto value = *(ptr + (ib*nx*ny1+idx_ipos1*ny1+is));

                        // slice_ftmp[ix] += coef[k] * fdist[ib][idx_ipos1][iz];
                        slice_ftmp[ix] += coef[k] * v;
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.async_work_group_copy(fdist.get_pointer()
                                        + g.get_group_id(2)
                                        + g.get_group_id(0) *ny1*nx, /* dest */
                                    slice_ftmp.get_pointer(), /* source */
                                    nx, /* n elems */
                                    ny1  /* stride */
            );
        });   // end parallel_for_work_group

#endif
    });       // end Q.submit
}
