add_subdirectory(advectorX)
add_subdirectory(advectorVx)

add_library(core "")

target_sources(core PUBLIC x_advectors.h IAdvector.h)
# target_sources(core PUBLIC x_advectors.h vx_advectors.h IAdvector.h)

target_link_libraries(core
  PUBLIC
  xadvec
  vxadvec
  adv1d::config
  Kokkos::kokkos
)

add_library(adv1d::core ALIAS core)
