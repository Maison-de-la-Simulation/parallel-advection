add_library(tools INTERFACE) #INTERFACE so we don't need source .cpp files

target_sources(tools
  PRIVATE
  init.h
  validation.h)

target_include_directories(tools
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR})

add_library(adv1d::tools ALIAS tools)
