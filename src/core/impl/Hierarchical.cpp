#include "advectors.h"

sycl::event
AdvX::Hierarchical::operator()(
    sycl::queue &Q,
    sycl::buffer<double, 2> &buff_fdistrib,
    const ADVParams &params) const
{
    auto const nx  = params.nx;
    auto const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

   const sycl::range<1> nb_wg{nVx};
   const sycl::range<1> wg_size{nx};

   return Q.submit([&](sycl::handler &cgh) {
       auto fdist =
           buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

       sycl::local_accessor<double, 1> slice_ftmp(sycl::range{nx}, cgh);

       cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<1> g) {
           g.parallel_for_work_item([&](sycl::h_item<1> it) {
               const int ix = it.get_local_id(0);
               const int ivx = g.get_group_id(0);

               double const xFootCoord = displ(ix, ivx, params);

               // Corresponds to the index of the cell to the left of
               // footCoord
               const int leftDiscreteCell =
                   sycl::floor((xFootCoord - minRealx) * inv_dx);

               const double d_prev1 =
                   LAG_OFFSET +
                   inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

               double coef[LAG_PTS];
               lag_basis(d_prev1, coef);

               const int ipos1 = leftDiscreteCell - LAG_OFFSET;
               double ftmp = 0.;

               slice_ftmp[ix] = 0;   // initializing slice for each work item
               for (int k = 0; k <= LAG_ORDER; k++) {
                   int idx_ipos1 = (nx + ipos1 + k) % nx;

                   slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
               }
           });   // end parallel_for_work_item --> Implicit barrier

           for (int i = 0; i < nx; ++i) {
               fdist[g.get_group_id(0)][i] = slice_ftmp[i];
           }
       });   // end parallel_for_work_group
   });       // end Q.submit
}