add_library(xadvec "")

target_sources(xadvec
  PRIVATE
  IAdvectorX.h
  impl/Sequential.cpp
  impl/BasicRange2D.cpp
  impl/BasicRange1D.cpp
  impl/Hierarchical.cpp
  impl/NDRange.cpp
  impl/Scoped.cpp
  impl/HierarchicalAlloca.cpp
  impl/FixedMemoryFootprint.cpp
)

target_include_directories(xadvec
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src/core/advectorX
  ) 

target_link_libraries(xadvec
  PUBLIC
  adv1d::config) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::xadvec ALIAS xadvec)
