// #include "advectors.h"

// sycl::event
// AdvX::BasicRange1D::operator()(sycl::queue &Q,
//                                sycl::buffer<double, 3> &buff_fdistrib,
//                                const ADVParams &params) {

//     auto const nx = params.nx;
//     auto const ny = params.ny;
//     auto const minRealX = params.minRealX;
//     auto const dx = params.dx;
//     auto const inv_dx = params.inv_dx;

//     Q.submit([&](sycl::handler &cgh) {
//         auto fdist =
//             buff_fdistrib.get_access<sycl::access::mode::read>(cgh);

//         /* Using the preallocated global buffer */
//         sycl::accessor ftmp(m_global_buff_ftmp, cgh, sycl::write_only,
//                             sycl::no_init);

//         cgh.parallel_for(sycl::range<1>(ny), [=](sycl::id<1> itm) {
//             const int iy = itm[0];

//             for (int ix = 0; ix < nx; ++ix) {
//                 double const xFootCoord = displ(ix, iy, params);

//                 // Corresponds to the index of the cell to the left of footCoord
//                 const int leftNode =
//                     sycl::floor((xFootCoord - minRealX) * inv_dx);

//                 const double d_prev1 =
//                     LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
//                                                               minRealX, dx));

//                 auto coef =  lag_basis(d_prev1);

//                 const int ipos1 = leftNode - LAG_OFFSET;

//                 ftmp[iy][ix] = 0;
//                 for (int k = 0; k <= LAG_ORDER; k++) {
//                     int idx_ipos1 = (nx + ipos1 + k) % nx;

//                     ftmp[iy][ix] += coef[k] * fdist[iy][idx_ipos1];

//                 }   // end for k
//             }       // end for ix
//             // barrier
//         });   // end parallel_for
//     });       // end Q.submit

//     return Q.submit([&](sycl::handler &cgh) {
//         auto fdist = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
//         auto ftmp =
//             m_global_buff_ftmp.get_access<sycl::access::mode::read>(cgh);
//         cgh.copy(ftmp, fdist);
//     });   // end Q.submit
// }
