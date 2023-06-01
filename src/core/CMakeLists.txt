add_subdirectory(advectorX)
# add_subdirectory(advectorVx)

add_library(core "")

target_sources(core PUBLIC advectors.h)

target_link_libraries(core
  PUBLIC
  xadvec
  # vxadvec
)

add_library(adv1d::core ALIAS core)
