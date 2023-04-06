#include "advectors.h"

sycl::event
AdvX::Scoped::operator()(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                         const ADVParams &params) const {
//   auto const nx = params.nx;
//   auto const nVx = params.nVx;
//   auto const minRealx = params.minRealx;
//   auto const dx = params.dx;
//   auto const inv_dx = params.inv_dx;

//   const sycl::range<1> nb_wg{nVx};
//   const sycl::range<1> wg_size{nx};

//   return Q.submit([&](sycl::handler &cgh) {
//     auto fdist = buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

//     cgh.parallel(nb_wg, wg_size, [=](auto g) {
//       // c.f.
//       // https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/scoped-parallelism.md#memory-placement-rules
//       double slice_ftmp[nx];   // declared in the private memory of the
//                                // executing physical WI ???

//       sycl::single_item_and_wait(g, [&]() {
//         for (int i = 0; i < nx; ++i) {
//           slice_ftmp[i] = fdist[g.get_group_id(0)][i];
//         }
//       });

//       sycl::distribute_groups_and_wait(g, [&](auto subg) {
//         sycl::distribute_items_and_wait(subg, [&](sycl::s_item<1> it) {
//           const int ix = it.get_local_id(g, 0);
//           const int ivx = g.get_group_id(0);

//           double const xFootCoord = displ(ix, ivx, params);

//           // Corresponds to the index of the cell to
//           // the left of footCoord
//           const int leftDiscreteCell =
//               sycl::floor((xFootCoord - minRealx) * inv_dx);

//           const double d_prev1 =
//               LAG_OFFSET +
//               inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

//           double coef[LAG_PTS];
//           lag_basis(d_prev1, coef);

//           const int ipos1 = leftDiscreteCell - LAG_OFFSET;
//           double ftmp = 0.;
//           for (int k = 0; k <= LAG_ORDER; k++) {
//             int idx_ipos1 = (nx + ipos1 + k) % nx;
//             ftmp += coef[k] * slice_ftmp[idx_ipos1];
//           }

//           fdist[ivx][ix] = ftmp;
//         });   // end distribute items
//       });     // end distribute_groups

//       // sycl::single_item_and_wait(g, [&](){
//       //    for(int i = 0; i < Nx ; ++i){
//       //       fdist[g.get_group_id(0)][i] = slice_ftmp[i];
//       //    }
//       // });
//     });   // end parallel_for_work_group
//   });     // end Q.submit
}