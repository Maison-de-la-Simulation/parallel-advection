add_library(config "")

target_sources(config
  PRIVATE
  ConfigMap.cpp
  inih/ini.cpp
  inih/INIReader.cpp
  AdvectionParams.cpp
  Conv1dParams.cpp)

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/config
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(config PUBLIC mdspan)
if(SYCL_IS_ACPP)
  add_sycl_to_target(TARGET config)
else()
    target_link_libraries(config PUBLIC dpcpp_opt)
endif()

#Add an alias so that library can be used inside the build tree
add_library(bkma::config ALIAS config)
