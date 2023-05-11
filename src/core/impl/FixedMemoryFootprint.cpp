#include "advectors.h"

sycl::event
AdvX::FixedMemoryFootprint::operator()(sycl::queue &Q,
                                       sycl::buffer<double, 2> &buff_fdistrib,
                                       const ADVParams &params) const noexcept {
    auto const nx = params.nx;
    auto const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    /* All this should be done in ctor not in kernel */
    auto const NB_SLICES_IN_MEMORY = 10;
    auto const NB_TOTAL_ITERATIONS = nVx / NB_SLICES_IN_MEMORY;
    auto const REST_ITERATIONS = nVx % NB_SLICES_IN_MEMORY;

    assert(REST_ITERATIONS == 0);   // for now nVx need to be divisible by NB_SLICES

    // sycl::buffer<double, 2> FTMP_BUFF{sycl::range<2>{NB_SLICES_IN_MEMORY, nx}};
    auto buff_ftmp = sycl::malloc_device<double>(nx * NB_SLICES_IN_MEMORY * sizeof(double), Q);

    sycl::free(buff_ftmp, Q);

    // const sycl::range<2> global_size{nVx, nx};
    // const sycl::range<2> local_size(1, nx);

    const sycl::range<2> nb_wg{NB_TOTAL_ITERATIONS, 1};
    const sycl::range<2> wg_size{1, 1024};

    //on regarde le reste de i avec NB_TOTAL_ITER et on divise

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<2> g) {
            g.parallel_for_work_item(
                sycl::range(1, nx), [&](sycl::h_item<2> item) {
                    const int ix = item.get_global_id(1);
                    const int ivx = item.get_global_id(0);

                    buff_ftmp[ix + nx * ivx] = fdist[ivx][ix];
                });

            g.parallel_for_work_item(
                sycl::range<2>(1, nx), [&](sycl::h_item<2> it) {
                    const int ix = it.get_global_id(1);
                    // g.get_group_id(0) also works for ivx
                    const int ivx = it.get_global_id(0);

                    double const xFootCoord = displ(ix, ivx, params);

                    // Corresponds to the index of the cell to the left of
                    // footCoord
                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(LeftDiscreteNode, minRealx, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    fdist[ivx][ix] = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;
                        fdist[ivx][ix] +=
                            coef[k] * buff_ftmp[idx_ipos1 + nx * ivx];
                    }
                });   // end parallel_for_work_item --> Implicit barrier
        });           // end parallel_for_work_group
    });               // end Q.submit
}