#include "IAdvectorX.h"
#include "advectors.h"

// this is streamY + straddlel malloc + twoDimwg
//  TODO: implement percent_in_global_mem values

constexpr size_t MAX_NX_ALLOC = 6144;   // A100
constexpr size_t MAX_NY = 128;

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::actual_advection(sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset,
                             const size_t &nx_rest_to_malloc) {

    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_y = params.wg_size_y;
    auto const wg_size_x = params.wg_size_x;

    /* ny must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_y != 0) {
        throw std::invalid_argument("ny_batch_size must be divisible by wg_size_y");
    }
    if (wg_size_y * nx > 6144) {
        std::cout << "wg_size_y = " << wg_size_y << ", nx = " << nx
                  << std::endl;
        throw std::invalid_argument(
            "wg_size_y*nx must be < to 6144 (shared memory limit)");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_y, 1, ny1};
    const sycl::range wg_size{params.wg_size_y, params.wg_size_x, 1};

    // std::cout << "flag1" << std::endl;
    // /* Straddled malloc*/
    // std::unique_ptr<sycl::buffer<double, 2>> ptr_buff_rest_nx;
    // // auto buff_rest_nx = *ptr_buff_rest_nx;
    // std::cout << "flag2" << std::endl;
    // if (nx_rest_to_malloc > 0)
    //     *ptr_buff_rest_nx = sycl::buffer<double, 2>(sycl::range{ny,
    //     nx_rest_to_malloc}, sycl::no_init);

    // std::cout << "flag3" << std::endl;
    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* We use a 2D local accessor here */
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            // if y1 > 1 //if we have a stide, we transpose, else we copy

            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iz = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int ivx =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[local_ny][ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[local_ny][ix] +=
                            coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iz = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int ivx =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    fdist[ivx][ix][iz] = slice_ftmp[local_ny][ix];
                });
        });   // end parallel_for_work_group
    });       // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp1::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;

    auto rest_malloc = nx <= MAX_NX_ALLOC ? 0 : nx - MAX_NX_ALLOC;

    // On A100 it breaks when ny (the first dimension) is >= 65536.
    if (ny < MAX_NY) {
        /* If limit not exceeded we return a classical Hierarchical advector */
        AdvX::Hierarchical adv{};
        return adv(Q, buff_fdistrib, params);
    } else {
        double div = static_cast<double>(ny) / static_cast<double>(MAX_NY);
        auto floor_div = std::floor(div);
        auto is_int = div == floor_div;
        auto n_batch = is_int ? div : floor_div + 1;

        for (int i_batch = 0; i_batch < n_batch - 1;
             ++i_batch) {   // can we parallel_for this on multiple GPUs?
                            // multiple queues ? or other CUDA streams?

            size_t ny_offset = (i_batch * MAX_NY);

            actual_advection(Q, buff_fdistrib, params, MAX_NY, ny_offset,
                             rest_malloc)
                .wait();
        }

        // for the last one we take the rest, we add n_batch-1 because we
        // processed MAX_SIZE-1 each batch
        auto const ny_size =
            is_int ? MAX_NY : (ny % MAX_NY);   // + (n_batch - 1);
        auto const ny_offset = MAX_NY * (n_batch - 1);

        // return the last advection with the rest
        return actual_advection(Q, buff_fdistrib, params, ny_size, ny_offset,
                                rest_malloc);
    }
}