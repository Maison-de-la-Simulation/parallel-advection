#include "advectors.h"

/* =================================================================
Coaliscing accesses + vertical distribution of data between local and
global memory: 2 types of kernels scheduled
==================================================================== */
void
print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
    if (lvl == 0)
        std::cout << "--------------------------------" << std::endl;
    std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
              << r.get(2) << "}" << std::endl;
}

// ==========================================
// ==========================================
sycl::event
AdvX::HybridMem::actual_advection(sycl::queue &Q, double *fdist_dev,
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

    sycl::range loc_range{k_local / loc_wg_size_0_, loc_wg_size_1_,
                          n2};   // TODO: bug here, not divisible

    const sycl::range glob_range{k_global / glob_wg_size_0_, glob_wg_size_1_,
                                 n2};   // TODO: bug here, not divisible

    const sycl::range glob_wgsize{glob_wg_size_0_, glob_wg_size_1_,
                                  glob_wg_size_2_};

    sycl::range loc_wgsize{loc_wg_size_0_, loc_wg_size_1_, loc_wg_size_2_};

    const size_t global_offset = k_local;
    // const size_t global_offset = 0;
    const auto ptr_global = scratchG_;

    auto const g_wg0 = glob_wg_size_0_;
    auto const g_wg1 = glob_wg_size_1_;
    auto const g_wg2 = glob_wg_size_2_;

    /* k_global: kernels running in the global memory, start from k_local to
     * last*/
    Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>{glob_range, glob_wgsize},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(ptr_global, k_global, n1, n2);

                const int i1 = itm.get_local_id(1);
                const int i2 = itm.get_global_id(2);

                const size_t k_n0 = itm.get_group().get_group_id(0);
                const int local_n0 = itm.get_local_id(0);
                const int i0 =
                    g_wg0 * k_n0 + n0_offset + local_n0 + global_offset;

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                for (int ii1 = i1; ii1 < n1; ii1 += g_wg1) {
                    scr(k_n0, ii1, i2) = solver(slice, i0, ii1, i2);
                }

                sycl::group_barrier(itm.get_group());

                for (int ii1 = i1; ii1 < n1; ii1 += g_wg1) {
                    fdist(i0, ii1, i2) = scr(k_n0, ii1, i2);
                }
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit

    /* SUBMIT LES NOYAUX RESTANT POUR LES KERNELS LOCAUX!!!!!!
    ON FAIT UN SUBMIT ET ON TRAITE EN MEME TEMPS QUE LE RESTE*/
    float div = static_cast<float>(n2) / static_cast<float>(loc_wg_size_2_);
    const size_t floor_div = std::floor(div);
    const auto div_is_int = div == floor_div;
    const auto rest_n2 = n2 % loc_wg_size_2_;

    /* If there is some rest, we schedule the kernels with uncontiguous access*/
    if (rest_n2 > 0) {
        /* Same sizes in dim 0 and 1, we only take the rest for dim 2 */
        const sycl::range rest_range{loc_range.get(0), loc_range.get(1),
                                     rest_n2};

        const sycl::range rest_wgsize{loc_wgsize.get(0), loc_wgsize.get(1),
                                      rest_n2 /*1*/};

        auto offset_rest_n2 = n2 - rest_n2;

        // std::cout << "My rest local accessor is of size: " << rest_n2 << "*"
        // << n1 << std::endl;

        Q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> slice_ftmp(
                sycl::range<2>(rest_n2, n1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>{rest_range, rest_wgsize},
                [=](auto itm) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const int i0 = itm.get_global_id(0);
                    const int i1 = itm.get_local_id(1);
                    const int i2 = itm.get_global_id(2);

                    const int i2_fdist = i2 + offset_rest_n2;

                    auto slice = std::experimental::submdspan(
                        fdist, i0, std::experimental::full_extent, i2);

                    for (int ii1 = i1; ii1 < n1; ii1 += rest_wgsize.get(1)) {
                        slice_ftmp[i2][ii1] = solver(slice, i0, ii1, i2_fdist);
                    }

                    sycl::group_barrier(itm.get_group());

                    for (int ii1 = i1; ii1 < n1; ii1 += rest_wgsize.get(1)) {
                        fdist(i0, ii1, i2_fdist) = slice_ftmp[i2][ii1];
                    }
                }   // end lambda in parallel_for
            );   // end parallel_for nd_range
        });      // end Q.submit

        //     /* Update the sizes of dim2 for the local kernels later, we */
        loc_wgsize =
            sycl::range{loc_wgsize.get(0), loc_wgsize.get(1), floor_div};
        loc_range = sycl::range{loc_range.get(0), loc_range.get(1),
                                floor_div * loc_wg_size_2_};

        // print_range("rest_range", rest_range, 0);
        // print_range("rest_wgsize", rest_wgsize, 1);
    }   // end if rest>0

    // print_range("loc_range", loc_range, 0);
    // print_range("loc_wgsize", loc_wgsize, 1);

    // print_range("glob_range", glob_range, 0);
    // print_range("glob_wgsize", glob_wgsize, 1);

    auto const l_wg1 = loc_wgsize.get(1);   // loc_wg_size_1_;
    auto const l_wg2 = loc_wgsize.get(2);   // loc_wg_size_2_;
    /* TODO: corect bug */

    // std::cout << "My local accessor is of size: " << l_wg2<< "*" << n1 <<
    // std::endl;

    /* Local kernels, start from 0 to k_local-1*/
    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(l_wg2, n1),
                                                   cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{loc_range, loc_wgsize},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                const int i0 = itm.get_global_id(0);
                const int i1 = itm.get_local_id(1);

                const int i2_local = itm.get_local_id(2);
                const int i2 =
                    i2_local +
                    itm.get_group().get_group_id(2) *
                        loc_wgsize.get(2) /*id_groupe*groupe_size TODO:*/;

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                for (int ii1 = i1; ii1 < n1; ii1 += l_wg1) {
                    slice_ftmp[i2_local][ii1] = solver(slice, i0, ii1, i2);
                }

                sycl::group_barrier(itm.get_group());

                for (int ii1 = i1; ii1 < n1; ii1 += l_wg1) {
                    fdist(i0, ii1, i2) = slice_ftmp[i2_local][ii1];
                }
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::HybridMem::operator()(sycl::queue &Q, double *fdist_dev,
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