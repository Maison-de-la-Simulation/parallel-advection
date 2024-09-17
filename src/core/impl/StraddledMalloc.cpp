#include "IAdvectorX.h"
#include "advectors.h"

constexpr size_t MAX_NX_ALLOC = 6144; //A100

// ==========================================
// ==========================================
sycl::event
AdvX::StraddledMalloc::adv_opt3(sycl::queue &Q,
                            sycl::buffer<double, 3> &buff_fdistrib,
                            const ADVParams &params,
                            const size_t &nx_rest_to_malloc) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{ny, 1, ny1};
    const sycl::range wg_size{1, params.wg_size_x, 1};

    sycl::buffer<double, 3> buff_rest_nx(sycl::range<3>{ny, nx_rest_to_malloc, ny1}, sycl::no_init);

    return Q.submit([&](sycl::handler &cgh) {

        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(MAX_NX_ALLOC), cgh);
        sycl::accessor overslice_ftmp(buff_rest_nx, cgh, sycl::read_write, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const auto ix = it.get_local_id(1);
                    const auto iy = g.get_group_id(0);
                    const auto iy1 = g.get_group_id(2);

                    //if ix > 6144; we use overslice_ftmp with index ix-MAX_NX_ALLOC
                    //else we use slice_ftmp

                    double const xFootCoord = displ(ix, iy, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    if(ix < MAX_NX_ALLOC){
                        slice_ftmp[ix] = 0.;
                        for (int k = 0; k <= LAG_ORDER; k++) {
                            int idx_ipos1 = (nx + ipos1 + k) % nx;

                            slice_ftmp[ix] += coef[k] * fdist[iy][idx_ipos1][iy1];
                        }
                    }
                    else{
                        overslice_ftmp[iy][ix-MAX_NX_ALLOC][iy1] = 0.;
                        for (int k = 0; k <= LAG_ORDER; k++) {
                            int idx_ipos1 = (nx + ipos1 + k) % nx;

                            overslice_ftmp[iy][ix-MAX_NX_ALLOC][iy1] += coef[k] * fdist[iy][idx_ipos1][iy1];
                        }

                    }
                }); // end parallel_for_work_item --> Implicit barrier

            g.parallel_for_work_item(sycl::range{1, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const auto ix = it.get_local_id(1);
                                         const auto iy = g.get_group_id(0);
                                         const auto iy1 = g.get_group_id(2);

                                        if(ix < MAX_NX_ALLOC)                                         
                                            fdist[iy][ix][iy1] = slice_ftmp[ix];
                                        else
                                            fdist[iy][ix][iy1] = overslice_ftmp[iy][ix-MAX_NX_ALLOC][iy1];
                                     });
        });   // end parallel_for_work_group
    });       // end Q.submit
} // end actual_advection


// ==========================================
// ==========================================
sycl::event
AdvX::StraddledMalloc::operator()(sycl::queue &Q,
                            sycl::buffer<double, 3> &buff_fdistrib,
                            const ADVParams &params) {
    size_t const nx = params.nx;
    // auto const ny = params.ny;
    // auto const ny1 = params.ny1;

    //On A100 it breaks if we allocate more than 48 KiB per block, which is 6144 double
    //On MI250x it breaks if we allocate more than 64KiB per wg, which is 8192 double
    if (nx <= MAX_NX_ALLOC) {
        // return adv_opt3(Q, buff_fdistrib, params, 0);
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        // cudaDeviceSynchronize(); //this works fine if CUDA backend
// cudaFuncSetAttribute(dynamicReverse,
//         cudaFuncAttributeMaxDynamicSharedMemorySize, S*sizeof(int) ));
//  cudaDeviceSynchronize();
//  cudaPeekAtLastError();

        //Option 2: use cudaFuncAttributeMaxDynamicSharedMemorySize
        //Option 3: use a mix of max local memory and global memory ??


        //=================================
        // Option 3
        // we could launch two kernels in parallel, one using shared mem and one using the rest in global mem
        auto rest_malloc = nx - MAX_NX_ALLOC;
        return adv_opt3(Q, buff_fdistrib, params, rest_malloc);
    }
}
