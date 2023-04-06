add_library(core "")

target_sources(core
  PRIVATE
  IAdvectorX.h
  impl/Sequential.cpp
  impl/BasicRange.cpp
  impl/Hierarchical.cpp
  impl/NDRange.cpp
)

target_include_directories(core
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src/config
  ) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::core ALIAS core)
