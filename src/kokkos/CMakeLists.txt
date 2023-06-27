add_subdirectory(core)
add_subdirectory(tools)

add_executable(advection-kokkos main.cpp)
target_link_libraries(advection-kokkos
    PUBLIC
    Kokkos::kokkos
    adv1d::config
    adv1d::core
    adv1d::tools)

target_include_directories(advection-kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})