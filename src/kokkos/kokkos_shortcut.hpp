#pragma once

#include <Kokkos_Core.hpp>
// #include <Kokkos_DualView.hpp>

using Kokkos::ALL;

using K_layout = Kokkos::LayoutRight;

using KV_double_1d=Kokkos::View<double*, K_layout>;
using KV_double_2d=Kokkos::View<double**, K_layout>;
using KV_double_3d=Kokkos::View<double***, K_layout>;

// using KVH_double_1d=KV_double_1d::HostMirror;
// using KVH_double_2d=KV_double_2d::HostMirror;
// using KVH_double_3d=KV_double_3d::HostMirror;

// using KV_cdouble_1d=Kokkos::View<const double*, K_layout>;
// using KV_cdouble_2d=Kokkos::View<const double**, K_layout>;
// using KV_cdouble_3d=Kokkos::View<const double***, K_layout>;
// using KV_cdouble_4d=Kokkos::View<const double****, K_layout>;
// using KV_cdouble_5d=Kokkos::View<const double*****, K_layout>;
// using KV_cdouble_6d=Kokkos::View<const double******, K_layout>;

// using KDV_double_1d=Kokkos::DualView<double*, K_layout>;
// using KDV_double_2d=Kokkos::DualView<double**, K_layout>;
// using KDV_double_3d=Kokkos::DualView<double***, K_layout>;
// using KDV_double_4d=Kokkos::DualView<double****, K_layout>;

// using KDV_cdouble_3d=Kokkos::DualView<const double***, K_layout>;
// using KDV_cdouble_4d=Kokkos::DualView<const double****, K_layout>;