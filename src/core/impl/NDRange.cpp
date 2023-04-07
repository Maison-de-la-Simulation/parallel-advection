#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          sycl::buffer<double, 2> &buff_fdistrib,
                          const ADVParams &params) const {
  auto const nx = params.nx;
  auto const nVx = params.nVx;
  auto const minRealx = params.minRealx;
  auto const dx = params.dx;
  auto const inv_dx = params.inv_dx;

  const sycl::range<2> global_size{nx, nVx};
  const sycl::range<2> local_size(1, 128);

  assert(nVx%128 == 0);

  return Q.submit([&](sycl::handler &cgh) {
    auto fdist = buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

    // sycl::local_accessor<double, 1> slice_ftmp(sycl::range{nx}, cgh);

    cgh.parallel_for(
        sycl::nd_range<2>{global_size, local_size},
        [=](sycl::nd_item<2> itm) {
          const int ix = itm.get_global_id(0);
          const int ivx = itm.get_local_id(1);

          double const xFootCoord = displ(ix, ivx, params);

          const int leftDiscreteCell =
              sycl::floor((xFootCoord - minRealx) * inv_dx);

          const double d_prev1 =
              LAG_OFFSET +
              inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

          double coef[LAG_PTS];
          lag_basis(d_prev1, coef);

          const int ipos1 = leftDiscreteCell - LAG_OFFSET;

          // slice_ftmp[ix] = 0;   // initializing slice for each work item
          double ftmp = 0.0;
          for (int k = 0; k <= LAG_ORDER; k++) {
            int idx_ipos1 = (nx + ipos1 + k) % nx;

            // slice_ftmp[ix] += coef[k] * fdist[idx_ipos1][ivx];
            ftmp += coef[k] * fdist[idx_ipos1][ivx];
          }

          sycl::group_barrier(itm.get_group());
          fdist[ix][ivx] = ftmp;
          // for (int i = 0; i < nx; ++i) {
          //   fdist[i][ivx] = slice_ftmp[i];
          // }
        }   // end lambda in parallel_for
    );      // end parallel_for nd_range
  });       // end Q.submit
}