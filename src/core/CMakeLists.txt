add_library(core)

target_sources(core
  PUBLIC
  BasicRange.cpp
  NDRange.cpp
  AdaptiveWg.cpp
  )

target_include_directories(core
  PUBLIC #need public to include "advectors.h" 
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/tools
  ${CMAKE_SOURCE_DIR}/src/config
  ${CMAKE_SOURCE_DIR}/src/solvers
)

target_link_libraries(core PUBLIC bkma::config mdspan)

if(SYCL_IS_ACPP)
  add_sycl_to_target(TARGET core)
else()
    target_link_libraries(core PUBLIC dpcpp_opt)
endif()

#Add an alias so that library can be used inside the build tree
add_library(bkma::core ALIAS core)
