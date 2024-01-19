#include "advectors.h"

sycl::event
AdvX::FakeAdvector::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) {
    auto const nx = params.nx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    return Q.submit([&](sycl::handler &cgh) {
        sycl::accessor fdist(buff_fdistrib, cgh, sycl::read_write);

        cgh.parallel_for(buff_fdistrib.get_range(), [=](sycl::id<2> itm) {
            const int ix = itm[1];
            const int ivx = itm[0];

            fdist[ivx][ix] += 1;
        });   // end parallel_for
    });       // end Q.submit
}

sycl::event
AdvX::FakeAdvector::stream_bench(sycl::queue &Q,
                                 sycl::buffer<double, 1> &buff) {

     return Q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc(buff, cgh, sycl::read_write);

        cgh.parallel_for(buff.get_range(), [=](sycl::id<1> itm){
            acc[itm] += 1;
        });
     });
}