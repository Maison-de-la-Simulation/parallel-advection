// #include "advectors.h"

// sycl::event
// AdvX::FakeAdvector::operator()(sycl::queue &Q,
//                                double* fdist_dev,
//                                const ADVParams &params) {
//     auto const n1 = params.n1;
//     auto const minRealX = params.minRealX;
//     auto const dx = params.dx;
//     auto const inv_dx = params.inv_dx;

//     return Q.submit([&](sycl::handler &cgh) {
//         sycl::accessor fdist(buff_fdistrib, cgh, sycl::read_write);

//         cgh.parallel_for(buff_fdistrib.get_range(), [=](sycl::id<2> itm) {
//             const int i1 = itm[1];
//             const int i0 = itm[0];

//             fdist[i0][i1] += 1;
//         });   // end parallel_for
//     });       // end Q.submit
// }

// sycl::event
// AdvX::FakeAdvector::stream_bench(sycl::queue &Q,
//                                  sycl::buffer<double, 1> &buff) {

//      return Q.submit([&](sycl::handler &cgh) {
//         sycl::accessor acc(buff, cgh, sycl::read_write);

//         cgh.parallel_for(buff.get_range(), [=](sycl::id<1> itm){
//             acc[itm] += 1;
//         });
//      });
// }