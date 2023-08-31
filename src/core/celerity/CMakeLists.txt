add_library(distr-core "")

target_sources(distr-core
  PRIVATE
  DistrAdvector.h 
  DistrAdvector.cpp
)

target_include_directories(core
  PUBLIC #need public so we can include "advectors.h" in impl/
  ${CMAKE_SOURCE_DIR}/src/core/celerity
  ${CMAKE_SOURCE_DIR}/src/config
  ) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::distr-core ALIAS distr-core)
add_celerity_to_target(TARGET distr-core SOURCES DistrAdvector.cpp)
