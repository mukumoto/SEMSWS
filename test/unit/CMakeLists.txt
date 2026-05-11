# =============================================================================
# Unit tests — standalone self-asserting C++ executables registered with CTest.
#
# Each test_* program returns 0 on success, non-zero on failure.
# =============================================================================

function(add_sem_unit_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE sem_srcrecv sem_config)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${name} PRIVATE cxx_std_17)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Helper that links against sem_fwi (and its transitive deps) for the AD
# sensitivity tests. Keeps the default helper above light for observed-data
# tests that don't need FWI symbols.
function(add_sem_fwi_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE sem_fwi sem_integ sem_material sem_config)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${name} PRIVATE cxx_std_17)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_sem_unit_test(observed_types_test)
add_sem_unit_test(observed_resampler_test)
add_sem_unit_test(hdf5_reader_catalog_test)
add_sem_unit_test(hdf5_reader_hyperslab_test)
add_sem_unit_test(hdf5_reader_compression_test)
add_sem_unit_test(hdf5_srcrecv_reader_basic_test)
add_sem_unit_test(hdf5_srcrecv_reader_sources_test)
add_sem_unit_test(dual_smoke_test)

# adjoint_source_operator_test — exercises AdjointSource::ApplyTimeOperator.
# Needs the whole runtime-linked chain (receivers + util + operator etc.).
add_executable(adjoint_source_operator_test adjoint_source_operator_test.cpp)
target_link_libraries(adjoint_source_operator_test PRIVATE sem)
target_compile_features(adjoint_source_operator_test PRIVATE cxx_std_17)
add_test(NAME adjoint_source_operator_test COMMAND adjoint_source_operator_test)

add_sem_fwi_test(acoustic_ad_vs_hand_test)
# 3D AD-vs-hand binaries are MPI-aware (Mpi::Init/Root inside main), so
# multi-rank runs are available by calling them directly:
#   mpirun -np 4 build/test/unit/acoustic_ad_vs_hand_3d_test
# CTest registers the serial invocation only, matching the 2D entry above.
add_sem_fwi_test(acoustic_ad_vs_hand_3d_test)
add_sem_fwi_test(elastic_ad_vs_hand_3d_test)

# MPI test — built like the others but registered once per rank count so
# `ctest` exercises nprocs=1,2,4 without relying on a particular launcher.
add_executable(hdf5_reader_mpi_test hdf5_reader_mpi_test.cpp)
target_link_libraries(hdf5_reader_mpi_test PRIVATE sem_srcrecv sem_config)
target_include_directories(hdf5_reader_mpi_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(hdf5_reader_mpi_test PRIVATE cxx_std_17)
foreach(np 1 2 4)
    add_test(NAME hdf5_reader_mpi_test_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE}
                     ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:hdf5_reader_mpi_test>
                     ${MPIEXEC_POSTFLAGS})
endforeach()
