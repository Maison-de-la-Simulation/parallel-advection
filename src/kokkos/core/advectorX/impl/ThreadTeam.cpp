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

    using team_member = typename Kokkos::TeamPolicy<>::member_type;

    // KV_double_3d ftmp("TMPARRAY_TO_BE_REMOVED", n_fict, nvx, nx);

    Kokkos::TeamPolicy<> policy(nvx * n_fict,
                                Kokkos::AUTO);   // not sure about AUTO
    // const Kokkos::TeamPolicy<> policy(nvx, nx);


    typedef Kokkos::DefaultExecutionSpace::scratch_memory_space
    ScratchSpace;
    // Define a view type in ScratchSpace
    typedef Kokkos::View<double*,ScratchSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_double_1d;

    // scr_t::shmem
    Kokkos::parallel_for(
        "advector::x::ThreadTeam::operator()::parallel_for",
        policy.set_scratch_size(0, Kokkos::PerTeam(nx)),
        KOKKOS_CLASS_LAMBDA(const team_member &team_h) {
            const auto i_fict = team_h.league_rank() % n_fict;
            const auto ivx = (team_h.league_rank() / n_fict);

            // scratch memory, 0 means shared memory, 1 means global mem
            shared_double_1d x_shared_slice(team_h.team_scratch(0), nx);

            // auto x_slice = Kokkos::subview(fdist, i_fict, ivx, Kokkos::ALL);
            // Kokkos::deep_copy(x_shared_slice, x_slice);

            //copy content into shared slice
            // Kokkos::parallel_for(
            //     Kokkos::TeamThreadRange(team_h, nx), [&](const int ix) {
            //         // x_shared_slice(ix) = 0;
            //         // x_shared_slice(ix) = fdist(i_fict, ivx, ix);
            //     }
            // );
            // team_h.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_h, nx), [&](const int ix) {
                    double const xFootCoord = displ(ix, ivx, params);

                    const int LeftDiscreteNode =
                        Kokkos::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(LeftDiscreteNode, minRealx, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    x_shared_slice(0) = 0;
                    // fdist(i_fict, ivx, ix) = 0;   // initializing slice
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        // fdist(i_fict, ivx, ix) +=
                            // coef[k] * x_shared_slice(idx_ipos1);
                    }

                    // std::cout << "ifict, ivx, ix= " << i_fict  << "," << ivx
                    // << ", " << ix << std::endl;
                });

            team_h.team_barrier();
        });

    // Kokkos::deep_copy(fdist, ftmp);
}