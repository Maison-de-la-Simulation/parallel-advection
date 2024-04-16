add_library(tools)

target_sources(tools
  PRIVATE
  unique_ref.h
  init.h
  init.cpp
  validation.h
  validation.cpp)

if(SYCL_IS_ACPP)
  add_sycl_to_target(TARGET tools)
else()
  target_link_libraries(tools PUBLIC dpcpp_opt)
endif()

target_link_libraries(tools PUBLIC adv1d::config adv1d::core)

add_library(adv1d::tools ALIAS tools)
