#include "advectors.h"

sycl::event
AdvX::Scoped::operator()(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                         const ADVParams &params) const {
  auto const nx = params.nx;
  auto const nVx = params.nVx;
  auto const minRealx = params.minRealx;
  auto const dx = params.dx;
  auto const inv_dx = params.inv_dx;

//   const sycl::range<2> nb_wg{1, nVx/4};
//   const sycl::range<2> wg_size{nx, 4};
  const sycl::range<2> nb_wg{2, nVx};
  const sycl::range<2> wg_size{nx/2, 1};

  return Q.submit([&](sycl::handler &cgh) {
    auto fdist = buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

    sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

    // cgh.copy()
    cgh.parallel(nb_wg, wg_size, [=](auto g) {
      // c.f.
      // https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/scoped-parallelism.md#memory-placement-rules
    //   double slice_ftmp[nx];   // declared in the private memory of the
                               // executing physical WI ???
        //Actually doesn't work if version of CUDA is not 11.6. I have to use the local_accessor


      sycl::single_item_and_wait(g, [&]() {
        for (int i = 0; i < nx; ++i) {
        //   slice_ftmp[i] = fdist[i][g.get_group_id(1)*4];
          slice_ftmp[i] = fdist[i][g.get_group_id(1)];
        }
      });

      sycl::distribute_groups_and_wait(g, [&](auto subg) {
        sycl::distribute_items_and_wait(subg, [&](sycl::s_item<2> it) {
          const int ix = it.get_local_id(g, 0);
          const int ivx = g.get_group_id(1);

          double const xFootCoord = displ(ix, ivx, params);

          // Corresponds to the index of the cell to
          // the left of footCoord
          const int leftDiscreteCell =
              sycl::floor((xFootCoord - minRealx) * inv_dx);

          const double d_prev1 =
              LAG_OFFSET +
              inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

          double coef[LAG_PTS];
          lag_basis(d_prev1, coef);

          const int ipos1 = leftDiscreteCell - LAG_OFFSET;
          double ftmp = 0.;
          for (int k = 0; k <= LAG_ORDER; k++) {
            int idx_ipos1 = (nx + ipos1 + k) % nx;
            ftmp += coef[k] * slice_ftmp[idx_ipos1];
          }

          fdist[ix][ivx] = ftmp;
        });   // end distribute items
      });     // end distribute_groups

      // sycl::single_item_and_wait(g, [&](){
      //    for(int i = 0; i < Nx ; ++i){
      //       fdist[g.get_group_id(0)][i] = slice_ftmp[i];
      //    }
      // });
    });   // end parallel_for_work_group
  });     // end Q.submit
}