#include "advectors.h"

sycl::event
AdvX::BasicRange1D::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) const {
  auto const nx = params.nx;
  auto const nVx = params.nVx;
  auto const minRealx = params.minRealx;
  auto const dx = params.dx;
  auto const inv_dx = params.inv_dx;

  return Q.submit([&](sycl::handler &cgh) {
    auto fdist = buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

    cgh.parallel_for(sycl::range<1>(nVx), [=](sycl::id<1> itm) {
      const int ivx = itm[0];
      // double slice_ftmp[Nx]; //static allocation
      double *slice_ftmp = (double *) alloca(nx * __SIZEOF_DOUBLE__);

      for (int ix = 0; ix < nx; ++ix) {
        double const xFootCoord = displ(ix, ivx, params);

        // Corresponds to the index of the cell to the left of footCoord
        const int leftDiscreteCell =
            sycl::floor((xFootCoord - minRealx) * inv_dx);

        const double d_prev1 =
            LAG_OFFSET +
            inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

        double *coef = (double *) alloca(LAG_PTS * __SIZEOF_DOUBLE__);
        // double coef[LAG_PTS];
        lag_basis(d_prev1, coef);

        const int ipos1 = leftDiscreteCell - LAG_OFFSET;

        slice_ftmp[ix] = 0;   // initializing slice for each work item
        for (int k = 0; k <= LAG_ORDER; k++) {
          int idx_ipos1 = (nx + ipos1 + k) % nx;

          slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
        }   // end for k
      }     // end for ix
      // barrier

      for (int i = 0; i < nx; ++i) {
        fdist[ivx][i] = slice_ftmp[i];
      }
    });   // end parallel_for
  });     // end Q.submit
}