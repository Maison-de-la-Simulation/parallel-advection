add_library(config "")

target_sources(config
  PRIVATE
  ConfigMap.cpp
  inih/ini.cpp
  inih/INIReader.cpp
  AdvectionParams.cpp
  InitParams.cpp
  unique_ref.h
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ) 

#Add an alias so that library can be used inside the build tree
add_library(adv1d::config ALIAS config)
