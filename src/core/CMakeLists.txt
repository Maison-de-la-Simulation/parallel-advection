add_library(core "")

target_sources(core
  PRIVATE
  IAdvectorX.h
  impl/Sequential.cpp
  impl/BasicRange2D.cpp
  impl/BasicRange1D.cpp
  impl/Hierarchical.cpp
  impl/NDRange.cpp
  impl/Scoped.cpp
)

target_include_directories(core
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src/config
  ) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::core ALIAS core)
