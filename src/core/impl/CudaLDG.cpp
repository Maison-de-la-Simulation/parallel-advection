#include "advectors.h"

#ifndef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
HIPSYCL_UNIVERSAL_TARGET
void optimized_codepaths(double* ptr, int ib, int n1, int n2, int id1_ipos, int is, double& value)
{
  __hipsycl_if_target_cuda(
        value = __ldg(ptr + (ib*n1*n2+id1_ipos*n2+is));
  );
  __hipsycl_if_target_host(
        value = *(ptr + (ib*n1*n2+id1_ipos*n2+is));
  );
}
#endif

sycl::event
AdvX::CudaLDG::operator()(sycl::queue &Q,
                          double* fdist_dev,
                          const ADVParams &params) {
    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{n0, 1, n2};
    const sycl::range wg_sise{1, params.wg_size_1, 1};
    // const sycl::range wg_sise{1, n1, 1};

    return Q.submit([&](sycl::handler &cgh) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
throw std::logic_error("CudaLDG kernel is not compatible with DPCPP");
#else   // for acpp

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_sise, [=](sycl::group<3> g) {
            // g.parallel_for_work_item(sycl::range{1, n1, 1},
            //                 [&](sycl::h_item<3> it) {
            //                     const int i1 = it.get_local_id(1);
            //                     const int ib = g.get_group_id(0);
            //                     const int is = g.get_group_id(2);
            //                     __ldg();
            //                     fdist[ib][i1][is] = slice_ftmp[i1];
            //                 });

            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);

                    const int i1 = it.get_local_id(1);
                    const int ib = g.get_group_id(0);
                    const int is = g.get_group_id(2);


                    double const xFootCoord = displ(i1, ib, params);

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
                    slice_ftmp[i1] = 0.;

                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int id1_ipos = (n1 + ipos1 + k) % n1;

                        double v = 0;
                        optimized_codepaths(fdist_dev, ib, n1, n2, id1_ipos, is, v);
                        // auto value = __ldg(ptr + (ib*n1*n2+id1_ipos*n2+is));
                        // double value = __ldg(&fdist[i0][id1_ipos][i2]);

                        // auto value = *(ptr + (ib*n1*n2+id1_ipos*n2+is));

                        // slice_ftmp[i1] += coef[k] * fdist[ib][id1_ipos][i2];
                        slice_ftmp[i1] += coef[k] * v;
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.async_work_group_copy(
                sycl::multi_ptr<double,
                                sycl::access::address_space::global_space>(
                    fdist_dev) +
                    g.get_group_id(2) + g.get_group_id(0) * n2 * n1, /* dest */
                slice_ftmp.get_pointer(), /* source */
                n1,                       /* n elems */
                n2                        /* stride */
            );
        });   // end parallel_for_work_group

#endif
    });       // end Q.submit
}
