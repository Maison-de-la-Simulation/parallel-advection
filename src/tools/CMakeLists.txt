add_library(tools)

target_sources(tools
  PRIVATE
  init.cpp
  validation.cpp)

if(SYCL_IS_ACPP)
  add_sycl_to_target(TARGET tools)
else()
  target_link_libraries(tools PUBLIC dpcpp_opt)
endif()

target_link_libraries(tools PUBLIC adv1d::config adv1d::core)
target_include_directories(tools
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/tools
  ${CMAKE_SOURCE_DIR}/src)

add_library(adv1d::tools ALIAS tools)
