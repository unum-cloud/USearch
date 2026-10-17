# What only the MPI drivers in distributed/ link: the MPI the cluster runs, never fetched, since a build must match the
# launcher and interconnect it starts under. CMake's `FindMPI` recognizes Open MPI, MPICH, Intel MPI and MS-MPI through
# their compiler wrappers, taking the first `mpicc` on the path; `-D MPI_C_COMPILER=<wrapper>` or `-D MPI_HOME=<prefix>`
# picks another where several are installed. Only the C interface is linked, so the library never sees MPI.
find_package(MPI REQUIRED COMPONENTS C)
