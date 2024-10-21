// #include "advectors.h"

// sycl::event
// AdvX::BasicRange1D::operator()(sycl::queue &Q,
//                                sycl::buffer<double, 3> &buff_fdistrib,
//                                const ADVParams &params) {

//     auto const n1 = params.n1;
//     auto const n0 = params.n0;
//     auto const minRealX = params.minRealX;
//     auto const dx = params.dx;
//     auto const inv_dx = params.inv_dx;

//     Q.submit([&](sycl::handler &cgh) {
//         auto fdist =
//             buff_fdistrib.get_access<sycl::access::mode::read>(cgh);

//         /* Using the preallocated global buffer */
//         sycl::accessor ftmp(m_global_buff_ftmp, cgh, sycl::write_only,
//                             sycl::no_init);

//         cgh.parallel_for(sycl::range<1>(n0), [=](sycl::id<1> itm) {
//             const int i0 = itm[0];

//             for (int i1 = 0; i1 < n1; ++i1) {
//                 double const xFootCoord = displ(i1, i0, params);

//                 // Corresponds to the index of the cell to the left of footCoord
//                 const int leftNode =
//                     sycl::floor((xFootCoord - minRealX) * inv_dx);

//                 const double d_prev1 =
//                     LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
//                                                               minRealX, dx));

//                 auto coef =  lag_basis(d_prev1);

//                 const int ipos1 = leftNode - LAG_OFFSET;

//                 ftmp[i0][i1] = 0;
//                 for (int k = 0; k <= LAG_ORDER; k++) {
//                     int id1_ipos = (n1 + ipos1 + k) % n1;

//                     ftmp[i0][i1] += coef[k] * fdist[i0][id1_ipos];

//                 }   // end for k
//             }       // end for i1
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
