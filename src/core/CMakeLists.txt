add_library(core)

target_sources(core
  PUBLIC
  IAdvectorX.h
  impl/BasicRange.cpp
  impl/NDRange.cpp
  impl/Hierarchical.cpp
  # impl/AdaptiveWg.cpp
  impl/HybridMem.cpp
  # impl/FullyLocal.cpp
  # impl/FullyGlobal.cpp
  )

target_include_directories(core
  PUBLIC #need public to include "advectors.h" in impl/*
  ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(core PUBLIC adv1d::tools adv1d::config mdspan)

  if(SYCL_IS_ACPP)
  add_sycl_to_target(TARGET core)
else()
    target_link_libraries(core PUBLIC dpcpp_opt)
endif()

#Add an alias so that library can be used inside the build tree
add_library(adv1d::core ALIAS core)
