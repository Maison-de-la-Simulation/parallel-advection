// #include "advectors.h"

// sycl::event
// AdvX::FixedMemoryFootprint::operator()(sycl::queue &Q,
//                                        sycl::buffer<double, 2> &buff_fdistrib,
//                                        const ADVParams &params) noexcept {
//     auto const n1 = params.n1;
//     auto const n0 = params.n0;
//     auto const minRealX = params.minRealX;
//     auto const dx = params.dx;
//     auto const inv_dx = params.inv_dx;

//     /* All this should be done in ctor not in kernel */
//     auto const NB_SLICES_IN_MEMORY = 10;
//     auto const NB_TOTAL_ITERATIONS = n0 / NB_SLICES_IN_MEMORY;
//     auto const REST_ITERATIONS = n0 % NB_SLICES_IN_MEMORY;

//     assert(REST_ITERATIONS ==
//            0);   // for now n0 need to be divisible by NB_SLICES

//     // sycl::buffer<double, 2> FTMP_BUFF{sycl::range<2>{NB_SLICES_IN_MEMORY,
//     // n1}};
//     auto buff_ftmp = sycl::malloc_device<double>(
//         n1 * NB_SLICES_IN_MEMORY * sizeof(double), Q);


//     // const sycl::range<2> global_size{n0, n1};
//     // const sycl::range<2> local_size(1, n1);

//     const sycl::range<2> nb_wg{NB_TOTAL_ITERATIONS, 1};
//     const sycl::range<2> wg_size{1, 1024};

//     // for (int n_iter = 0; n_iter < NB_TOTAL_ITERATIONS; ++n_iter) {

//     //     auto first_row_number = n_iter * NB_SLICES_IN_MEMORY;

//     //     const sycl::range<2> nb_wg{NB_SLICES_IN_MEMORY, 1};
//     //     const sycl::range<2> wg_size{1, n1};

//     //     Q.submit([&](sycl::handler &cgh) {
//     //         auto fdist =
//     //             buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);
//     //         cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<2> g) {
//     //             auto row_number = first_row_number + g.get_group_id(0);   // where we should take the slice in memory

//     //             // copying slice
//     //             g.parallel_for_work_item(
//     //                 sycl::range(1, n1), [&](sycl::h_item<2> item) {
//     //                     const int i1 = item.get_global_id(1);
//     //                     const int i0 =
//     //                         item.get_global_id(0) + first_row_number;

//     //                     buff_ftmp[i1 + n1 * row_number] = fdist[i0][i1];
//     //                 });

//     //             g.parallel_for_work_item(
//     //                 sycl::range<2>(1, n1), [&](sycl::h_item<2> it) {
//     //                     const int i1 = it.get_global_id(1);
//     //                     const int i0 = it.get_global_id(0) + first_row_number;

//     //                     double const xFootCoord = displ(i1, i0, params);

//     //                     // Corresponds to the index of the cell to the left of
//     //                     // footCoord
//     //                     const int leftNode =
//     //                         sycl::floor((xFootCoord - minRealX) * inv_dx);

//     //                     const double d_prev1 =
//     //                         LAG_OFFSET +
//     //                         inv_dx * (xFootCoord -
//     //                                   coord(leftNode, minRealX, dx));

//     //                     auto coef = lag_basis(d_prev1);

//     //                     const int ipos1 = leftNode - LAG_OFFSET;

//     //                     fdist[i0][i1] = 0;
//     //                     for (int k = 0; k <= LAG_ORDER; k++) {
//     //                         int id1_ipos = (n1 + ipos1 + k) % n1;
//     //                         fdist[i0][i1] +=
//     //                             coef[k] * buff_ftmp[id1_ipos + n1 * row_number];
//     //                     }
//     //                 });   // end parallel_for_work_item --> Implicit barrier
//     //         });           // end parallel_for_work_group
//     //     }).wait();               // end Q.submit
//     // }

//     // sycl::free(buff_ftmp, Q);

//     return Q.submit([&](sycl::handler &cgh) {
//         // cgh.single_task([=]() {
//         // });  
//     });   
// }
