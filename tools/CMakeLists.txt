# The command-line tools, calling the library through its C interface: `usearch` runs every verb on one process, and
# `usearch-mpi` runs them across an MPI world. Flags parse through the table the benchmarks share. MPI is the one the
# cluster runs, never fetched, since a build must match the launcher and interconnect it starts under: `FindMPI` takes
# the first `mpicc` on the path, and `-D MPI_C_COMPILER=<wrapper>` or `-D MPI_HOME=<prefix>` picks another.
add_executable(usearch_tools usearch.cpp dense.cpp checkpoints.cpp world_local.cpp)
usearch_setup_target(usearch_tools HOST)
set_target_properties(usearch_tools PROPERTIES OUTPUT_NAME usearch)
target_include_directories(usearch_tools PRIVATE ${PROJECT_SOURCE_DIR}/bench)
target_link_libraries(usearch_tools PRIVATE ${USEARCH_PUNNED_LIBRARIES} fmt::fmt forkunion::static yyjson)
install(TARGETS usearch_tools RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tools)

find_package(
    MPI
    COMPONENTS C
    QUIET
)
if (MPI_C_FOUND)
    add_executable(usearch_tools_mpi usearch.cpp dense.cpp checkpoints.cpp world_mpi.cpp)
    usearch_setup_target(usearch_tools_mpi HOST)
    set_target_properties(usearch_tools_mpi PROPERTIES OUTPUT_NAME usearch-mpi)
    target_include_directories(usearch_tools_mpi PRIVATE ${PROJECT_SOURCE_DIR}/bench)
    target_link_libraries(
        usearch_tools_mpi PRIVATE ${USEARCH_PUNNED_LIBRARIES} fmt::fmt forkunion::static yyjson MPI::MPI_C
    )
    install(TARGETS usearch_tools_mpi RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tools)
endif ()
