add_subdirectory(advectorX)
add_subdirectory(advectorVx)

add_library(core "")

target_sources(core PUBLIC advectors.h IAdvector.h)

target_link_libraries(core
  PUBLIC
  xadvec
  vxadvec
  adv1d::config
)

# target_include_directories(xadvec
#   PUBLIC #need public so we can include "advectors.h" in impl/
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/advectorX
#   ${CMAKE_CURRENT_SOURCE_DIR}/advectorVx
#   ) 


add_library(adv1d::core ALIAS core)
