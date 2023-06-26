add_subdirectory(config)

if("${BACKEND}" STREQUAL "sycl")
    add_subdirectory(sycl)
endif()
if("${BACKEND}" STREQUAL "kokkos")
    add_subdirectory(kokkos)
endif()

configure_file(advection.ini ${BIN_DIR}/advection.ini COPYONLY)