add_library(xadvec "")

target_sources(xadvec
  PRIVATE
  IAdvectorX.h
  impl/MDRange.cpp
)

target_include_directories(xadvec
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/kokkos/core
  ${CMAKE_CURRENT_SOURCE_DIR}
  ) 

target_link_libraries(xadvec
  PUBLIC
  adv1d::config
  Kokkos::kokkos)

#Add an alias so that library can be used inside the build tree
add_library(adv1d::xadvec ALIAS xadvec)
