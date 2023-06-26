add_library(vxadvec "")

target_sources(vxadvec
  PRIVATE
  IAdvectorVx.h
  impl/Hierarchical.cpp
)

target_include_directories(vxadvec
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/sycl/core
  ${CMAKE_CURRENT_SOURCE_DIR}
  ) 

target_link_libraries(vxadvec
  PUBLIC
  adv1d::config) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::vxadvec ALIAS vxadvec)