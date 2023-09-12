#include "advectors.h"

sycl::event
AdvX::Sequential::operator()(sycl::queue &Q,
                             sycl::buffer<double, 2> &buff_fdistrib,
                             const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist_write =
            buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
        auto fdist_read =
            buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
        // auto fdist_p1    =
        // buff_fdistrib_p1.get_access<sycl::access::mode::write>(cgh);

        cgh.single_task([=]() {
            // For each Vx
            for (int ivx = 0; ivx < nvx; ++ivx) {

                // std::array<double, Nx> ftmp{};
                // double* x_slice = sycl::malloc_device(sizeof(double)*Nx,);

                // Problem here is that with thie method
                // we have to set the accessor in read_write mode instead of 2
                // accessors, one in read, one in write
                double slice_x[nx];
                memcpy(slice_x, fdist_read.get_pointer() + ivx * nx,
                       nx * sizeof(double));

                // For each x with regards to current Vx
                for (int ix = 0; ix < nx; ++ix) {
                    double const xFootCoord = displ(ix, ivx, params);

                    // Corresponds to the index of the cell to the left of
                    // footCoord
                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    // d_prev1 : dist entre premier point utilisé pour
                    // l'interpolation et xFootCoord (dans l'espace de coord
                    // discret, même si double)

                    /* Percentage of the distance inside the cell ???? TODO :
                     * Find better var name */
                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx *
                            (xFootCoord - (minRealX + LeftDiscreteNode * dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;
                    double ftmp = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 =
                            (nx + ipos1 + k) %
                            nx;   // penser à essayer de retirer ce modulo.
                                  // Possible si on a une distance max on alloue
                                  // un tableau avec cette distance max en plus
                                  // des deux côtés

                        // Pour faire in place il faut utiliser un buffer de
                        // taille Nx. Soit on s'en sert en buffer d'input pour
                        // la lecture et on met la valeur directement dans fdist
                        //  soit on s'en sert en buffer output : on stocke le
                        //  résultat dedans puis on copie tout ce buffer dans la
                        //  ligne correspondante dans fdist

                        /* Ici en utilisant fdist en lecture */
                        // ftmp += coef[k] * fdist[ivx][idx_ipos1];

                        /* Ici en utilisant slice_x as an input */
                        ftmp += coef[k] * slice_x[idx_ipos1];

                        /*  */
                        // ftmp[idx_pos1] += coef[k] * fdist[ivx][idx_ipos1];
                    }

                    // fdist_write[ivx][ix] = ftmp;
                    fdist_write[ivx][ix] = ftmp;
                }   // end for X

            }       // end for Vx
        });         // end cgh.single_task()
    });             // end Q.submit
}