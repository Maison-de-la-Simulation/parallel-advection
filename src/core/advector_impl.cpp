#include "advector_impl.h"

sycl::event
AdvectorX::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) const {
  auto const nx = params.nx;
  auto const nVx = params.nVx;
  auto const minRealx = params.minRealx;
  auto const dx = params.dx;
  auto const inv_dx = params.inv_dx;

  /* Cannot use local memory with basic range parallel_for so I use a global
  buffer of size NVx * Nx*/
  sycl::buffer<double, 2> global_buff_ftmp(sycl::range<2>(nVx, nx));

  Q.submit([&](sycl::handler &cgh) {
    auto fdist = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);

    sycl::accessor ftmp(global_buff_ftmp, cgh, sycl::write_only, sycl::no_init);

    cgh.parallel_for(buff_fdistrib.get_range(), [=](sycl::id<2> itm) {
      const int ix = itm[1];
      const int ivx = itm[0];

      double const xFootCoord = displ(ix, ivx, params);

      // Corresponds to the index of the cell to the left of footCoord
      const int leftDiscreteCell =
          sycl::floor((xFootCoord - minRealx) * inv_dx);

      const double d_prev1 =
          LAG_OFFSET +
          inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

      double coef[LAG_PTS];
      lag_basis(d_prev1, coef);

      const int ipos1 = leftDiscreteCell - LAG_OFFSET;

      ftmp[ivx][ix] = 0;   // initializing slice for each work item
      for (int k = 0; k <= LAG_ORDER; k++) {
        int idx_ipos1 = (nx + ipos1 + k) % nx;

        ftmp[ivx][ix] += coef[k] * fdist[ivx][idx_ipos1];
      }

      // barrier
    });   // end parallel_for
  });     // end Q.submit

  // With basic range I have to submit 2 kernels in order to have a barrier
  // this means I cannot use a local accessor in the previous kernel
  return Q.submit([&](sycl::handler &cgh) {
    auto fdist = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
    auto ftmp = global_buff_ftmp.get_access<sycl::access::mode::read>(cgh);
    cgh.copy(ftmp, fdist);

    // cgh.parallel_for(sycl::range<1>(nVx), [=](sycl::id<1> itm) {
    //     const int ivx = itm[0];

    //     for (int i = 0; i < nx; ++i) {
    //         fdist[ivx][i] = ftmp[ivx][i];
    //     }
    // });   // end parallel_for
  });   // end Q.submit
}