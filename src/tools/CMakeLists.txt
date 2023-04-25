add_library(tools INTERFACE) #INTERFACE so we don't need source .cpp files

target_sources(tools
  PRIVATE
  init.h
  unique_ref.h
  validation.h)

target_include_directories(core
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/tools)

add_library(adv1d::tools ALIAS tools)
