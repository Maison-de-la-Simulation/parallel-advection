add_library(config "")

target_sources(config
  PRIVATE
  ConfigMap.cpp
  inih/ini.cpp
  inih/INIReader.cpp
  AdvectionParams.cpp
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/config
  ) 

target_link_libraries(config PUBLIC compiler_wrapper)

#Add an alias so that library can be used inside the build tree
add_library(adv1d::config ALIAS config)
