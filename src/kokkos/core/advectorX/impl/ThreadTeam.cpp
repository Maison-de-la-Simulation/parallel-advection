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

    using team_member = typename Kokkos::TeamPolicy<>::member_type;

    Kokkos::TeamPolicy<> policy(nvx * n_fict,
                                Kokkos::AUTO); // not sure about AUTO

    // Define a view type in ScratchSpace
    typedef Kokkos::View<double*,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            ScratchViewType;
    
    //this line is actually allocating shared mem, segfault if no shmem_size 
    int scratch_size = ScratchViewType::shmem_size(nx);

    Kokkos::parallel_for(
        "advector::x::ThreadTeam::operator()::parallel_for",
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_CLASS_LAMBDA(const team_member &team_h) {
            const auto i_fict = team_h.league_rank() % n_fict;
            const auto ivx    = (team_h.league_rank() / n_fict);

            // scratch memory, 0 means shared memory, 1 means global mem
            ScratchViewType x_shared_slice(team_h.team_scratch(0), nx);

            // auto x_slice = Kokkos::subview(fdist, i_fict, ivx, Kokkos::ALL);
            // Kokkos::deep_copy(x_shared_slice, x_slice);

            //copy content into shared slice
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_h, nx), [&](const int ix) {
                    x_shared_slice(ix) = fdist(i_fict, ivx, ix);
                }
            );
            team_h.team_barrier();

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

                    fdist(i_fict, ivx, ix) = 0;   // initializing slice
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        fdist(i_fict, ivx, ix) +=
                            coef[k] * x_shared_slice(idx_ipos1);
                    }
                });
        });
}