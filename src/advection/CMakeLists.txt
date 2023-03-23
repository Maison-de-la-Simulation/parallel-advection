add_library(core "")

target_sources(core
  PRIVATE
  i_advector_x.h
  advector_impl.cpp
)

target_include_directories(core
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/advection
  ${CMAKE_SOURCE_DIR}/src/config
  ) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::core ALIAS core)
