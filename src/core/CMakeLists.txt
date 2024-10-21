add_library(core)

target_sources(core
  PUBLIC
  IAdvectorX.h
  impl/Sequential.cpp
  impl/BasicRange.cpp
  # impl/BasicRange1D.cpp
  impl/Hierarchical.cpp
  impl/NDRange.cpp
  impl/Scoped.cpp
  # impl/FakeAdvector.cpp
  # impl/HierarchicalAlloca.cpp
  # impl/FixedMemoryFootprint.cpp
  impl/StreamY.cpp
  # impl/ReducedPrecision.cpp
  impl/StraddledMalloc.cpp
  impl/ReverseIndexes.cpp
  impl/TwoDimWG.cpp
  impl/SeqTwoDimWG.cpp
  impl/Exp1.cpp
  impl/Exp2.cpp
  impl/Exp3.cpp
  impl/Exp4.cpp
  impl/Exp5.cpp
  impl/CudaLDG.cpp
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
