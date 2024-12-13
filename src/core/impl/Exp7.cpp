#include "advectors.h"

/* =================================================================
Coaliscing accesses + vertical distribution of data between local and
global memory: 2 types of kernels scheduled
==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp7::actual_advection(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver, const size_t &n0_batch_size,
                             const size_t &n0_offset, const size_t k_global,
                             const size_t k_local) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    /* n0 must be divisible by slice_size_dim_y */
    if (n0_batch_size % loc_wg_size_0_ != 0 ||
        n0_batch_size % glob_wg_size_0_ != 0) {
        throw std::invalid_argument(
            "n0_batch_size must be divisible by [loc/glob]_wg_size_0");
    }
    if (loc_wg_size_0_ * n1 > MAX_LOCAL_ALLOC_) {
        throw std::invalid_argument("loc_wg_size_0_*n1 must be < to "
                                    "MAX_LOCAL_ALLOC_ (local memory limit)");
    }

    const sycl::range nb_wg_local{k_local/loc_wg_size_0_,
                                  1,
                                  n2/loc_wg_size_2_}; //TODO: bug here, not divisible

    const sycl::range nb_wg_global{k_global/glob_wg_size_0_,
                                   1,
                                   n2/glob_wg_size_2_}; //TODO: bug here, not divisible

    std::cout << "local ndrange : {" << nb_wg_local.get(0) << ","
              << nb_wg_local.get(1) << "," << nb_wg_local.get(2)
              << "}" << std::endl;

    std::cout << "global ndrange: {" << nb_wg_global.get(0) << ","
              << nb_wg_global.get(1) << "," << nb_wg_global.get(2)
              << "}" << std::endl;

    const sycl::range glob_wg_size{glob_wg_size_0_, glob_wg_size_1_,
                                   glob_wg_size_2_};

    const sycl::range loc_wg_size{loc_wg_size_0_  , loc_wg_size_1_, loc_wg_size_2_};

    const size_t global_offset = k_local;
    const auto ptr_global = scratchG_;

    auto const g_wg0 = glob_wg_size_0_;
    auto const g_wg1 = glob_wg_size_1_;
    auto const g_wg2 = glob_wg_size_2_;

    auto const l_wg0 = loc_wg_size_0_;
    auto const l_wg1 = loc_wg_size_1_;
    auto const l_wg2 = loc_wg_size_2_;

    /* k_global: kernels running in the global memory */
    Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>{nb_wg_global, glob_wg_size},
                         [=](auto itm) {
                             mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                             mdspan3d_t scr(ptr_global, k_global, n1,
                                                     n2);

                             const int i1 = itm.get_local_id(1);
                             const int i2 = itm.get_global_id(2);

                             const size_t k_n0 = itm.get_group().get_group_id(0);
                             const int local_n0 = itm.get_local_id(0);
                             const int i0 = g_wg0 * k_n0 + n0_offset +
                                            local_n0 + global_offset;

                             auto slice = std::experimental::submdspan(
                                 fdist, i0, std::experimental::full_extent, i2);

                             for (int ii1 = i1; ii1 < n1; ii1 += g_wg1) {
                                 scr(i0, ii1, i2) = solver(slice, i0, ii1, i2);
                             }

                             sycl::group_barrier(itm.get_group());

                             for (int ii1 = i1; ii1 < n1; ii1 += g_wg1) {
                                 fdist(i0, ii1, i2) = scr(i0, ii1, i2);
                             }
                         }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(l_wg2, n1),
                                                   cgh);

        cgh.parallel_for(sycl::nd_range<3>{nb_wg_local, loc_wg_size},
                         [=](auto itm) {
                             mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                             const int i0 = itm.get_global_id(0);
                             const int i1 = itm.get_local_id(1);
                             const int i2 = itm.get_global_id(2);

                             auto slice = std::experimental::submdspan(
                                 fdist, i0, std::experimental::full_extent, i2);

                             for (int ii1 = i1; ii1 < n1; ii1 += l_wg1) {
                                 slice_ftmp[i2][ii1] =
                                     solver(slice, i0, ii1, i2);
                             }
                             // }

                             sycl::group_barrier(itm.get_group());

                             for (int ii1 = i1; ii1 < n1; ii1 += l_wg1) {
                                 fdist(i0, ii1, i2) = slice_ftmp[i2][ii1];
                             }
                         }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp7::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {

    // can be parallel on multiple streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {

        size_t n0_offset = (i_batch * MAX_N0_BATCHS_);

        actual_advection(Q, fdist_dev, solver, MAX_N0_BATCHS_, n0_offset,
                         k_global_, k_local_)
            .wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, solver, last_n0_size_,
                            last_n0_offset_, last_k_global_, last_k_local_);
}