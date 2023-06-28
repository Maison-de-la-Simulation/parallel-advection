#include "x_advectors.h"

void
advector::x::ThreadTeam::operator()(KV_double_3d &fdist,
                                    const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const n_fict = params.n_fict_dim;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    // const Kokkos::Array<int, 3> begin{0, 0, 0};
    // const Kokkos::Array<int, 3> end{fdist.extent_int(0), fdist.extent_int(1),
    //                                 fdist.extent_int(2)};

    // Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy(begin, end);

    using Kokkos::parallel_reduce;
    using team_member = typename Kokkos::TeamPolicy<>::member_type;

    const Kokkos::TeamPolicy<> policy(nvx*n_fict, Kokkos::AUTO); // not sure about AUTO
    // const Kokkos::TeamPolicy<> policy(nvx, nx);

    Kokkos::parallel_for(
        "advector::x::ThreadTeam::operator()::parallel_for", policy,
        KOKKOS_CLASS_LAMBDA(const team_member &team_h) {
            const auto i_fict = team_h.league_rank() % n_fict;
            // const auto i_fict = team_h.league_rank() % nvx;
            const auto ivx = (team_h.league_rank() / n_fict);
            // Kokkos::floor(team_h.team_rank() / n_fict);


            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_h, nx),
                                 [&] (const int ix){
                                    std::cout << "ifict, ivx, ix= " << i_fict  << "," << ivx << ", " << ix << std::endl; 
                                 });

            // team_h.team_barrier();
        });

    // This is a reduction with a team policy.  The team policy changes
    // the first argument of the lambda.  Rather than an integer index
    // (as with RangePolicy), it's now TeamPolicy::member_type.  This
    // object provides all information to identify a thread uniquely.
    // It also provides some team-related function calls such as a team
    // barrier (which a subsequent example will use).
    //
    // Every member of the team contributes to the total sum.  It is
    // helpful to think of the lambda's body as a "team parallel
    // region."  That is, every team member is active and will execute
    // the body of the lambda.
    //     int sum = 0;
    // // We also need to protect the usage of a lambda against compiling
    // // with a backend which doesn't support it (i.e. Cuda 6.5/7.0).
    // #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    //     parallel_reduce(
    //         policy,
    //         KOKKOS_LAMBDA(const team_member &thread, int &lsum) {
    //             lsum += 1;
    //         // TeamPolicy<>::member_type provides functions to query the
    //         // multidimensional index of a thread, as well as the number of
    //         // thread teams and the size of each team.
    // #ifndef __SYCL_DEVICE_ONLY__
    //             // FIXME_SYCL needs workaround for printf
    //             printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
    //                    thread.team_rank(), thread.league_size(),
    //                    thread.team_size());
    // #else
    //             (void) thread;
    // #endif
    //         },
    //         sum);
    // #endif
    // The result will be 12*team_policy::team_size_max([=]{})
    // printf("Result %i\n", sum);

    // Kokkos::parallel_for(
    //     "MDrange_advectionX", mdrange_policy,
    //     KOKKOS_CLASS_LAMBDA(int i, int j, int k) {
    //         const auto i_fict = i;
    //         const auto ivx = j;
    //         const auto ix = k;

    //         double const xFootCoord = displ(ix, ivx, params);

    //         // const double x = params.minRealx + ix * params.dx;

    //         // Corresponds to the index of the cell to the left of footCoord
    //         const int LeftDiscreteNode =
    //             Kokkos::floor((xFootCoord - minRealx) * inv_dx);

    //         const double d_prev1 =
    //             LAG_OFFSET +
    //             inv_dx * (xFootCoord - coord(LeftDiscreteNode, minRealx,
    //             dx));

    //         auto coef = lag_basis(d_prev1);

    //         const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

    //         m_ftmp(i_fict, ivx, ix) = 0;   // initializing slice
    //         for (int k = 0; k <= LAG_ORDER; k++) {
    //             int idx_ipos1 = (nx + ipos1 + k) % nx;

    //             m_ftmp(i_fict, ivx, ix) +=
    //                 coef[k] * fdist(i_fict, ivx, idx_ipos1);
    //         }
    //     });

    // Kokkos::deep_copy(fdist, m_ftmp);
}