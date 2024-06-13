#include "IAdvectorX.h"
#include "advectors.h"

constexpr size_t MAX_NX_ALLOC = 6144; //A100

// AdvX::StraddledMalloc::StraddledMalloc(const ADVParams &params){
//     auto n_batch = std::ceil(params.nb / MAX_nb); //should be in
//     constructor

// }

// ==========================================
// ==========================================
sycl::event
AdvX::StraddledMalloc::adv_opt3(sycl::queue &Q,
                            sycl::buffer<double, 3> &buff_fdistrib,
                            const ADVParams &params,
                            const size_t &nx_rest_to_malloc) {
    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{nb, 1, ns};
    const sycl::range wg_size{1, params.wg_size_x, 1};

    //TODO we don't want this, we want to allocate a 1D slice for each problem in parallel, containing only the rest of NX slice in 1D
    sycl::buffer<double, 3> buff_rest_nx(sycl::range<3>{nb, nx_rest_to_malloc, ns}, sycl::no_init);

    return Q.submit([&](sycl::handler &cgh) {

        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(MAX_NX_ALLOC), cgh);
        sycl::accessor overslice_ftmp(buff_rest_nx, cgh, sycl::read_write, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int ivx = g.get_group_id(0);
                    const int iz = g.get_group_id(2);

                    //if ix > 6144; we use overslice_ftmp with index ix-MAX_NX_ALLOC
                    //else we use slice_ftmp

                    double const xFootCoord = displ(ix, ivx, params);

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

                            slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1][iz];
                        }
                    }
                    else{
                        overslice_ftmp[ivx][ix-MAX_NX_ALLOC][iz] = 0.;
                        for (int k = 0; k <= LAG_ORDER; k++) {
                            int idx_ipos1 = (nx + ipos1 + k) % nx;

                            overslice_ftmp[ivx][ix-MAX_NX_ALLOC][iz] += coef[k] * fdist[ivx][idx_ipos1][iz];
                        }

                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.parallel_for_work_item(sycl::range{1, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int ivx = g.get_group_id(0);
                                         const int iz = g.get_group_id(2);

                                        if(ix < MAX_NX_ALLOC)                                         
                                            fdist[ivx][ix][iz] = slice_ftmp[ix];
                                        else
                                            fdist[ivx][ix][iz] = overslice_ftmp[ivx][ix-MAX_NX_ALLOC][iz];
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
    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;

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

        //Option 1: use global memory --> Out of place
        //Option 2: use cudaFuncAttributeMaxDynamicSharedMemorySize
        //Option 3: use a mix of max local memory and global memory ??


        //=================================
        // Option 3
        // we could launch two kernels in parallel, one using shared mem and one using the rest in global mem
        auto rest_malloc = nx - MAX_NX_ALLOC;
        return adv_opt3(Q, buff_fdistrib, params, rest_malloc);
    }
}
