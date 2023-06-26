add_library(xadvec "")

target_sources(xadvec
  PRIVATE
  IAdvectorX.h
  impl/Sequential.cpp
  impl/BasicRange3D.cpp
  impl/BasicRange1D.cpp
  impl/Hierarchical.cpp
  impl/NDRange.cpp
  impl/Scoped.cpp
  impl/HierarchicalAlloca.cpp
  impl/FixedMemoryFootprint.cpp
)

target_include_directories(xadvec
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/sycl/core
  ${CMAKE_CURRENT_SOURCE_DIR}
  ) 

target_link_libraries(xadvec
  PUBLIC
  adv1d::config) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::xadvec ALIAS xadvec)
