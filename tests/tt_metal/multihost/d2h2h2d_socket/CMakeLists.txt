# One leg per target, each a single rank with no peer -- so nothing here needs an MPI
# launcher, even though it lives under multihost/.
add_library(d2h2h2d_socket_d2h_bw_test OBJECT)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_d2h_bw_test)

target_sources(d2h2h2d_socket_d2h_bw_test PRIVATE test_d2h_bw.cpp)

target_include_directories(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
)

# TT_DIRECT_KERNEL_DIR: the test passes an absolute path to CreateKernel, so the kernel is
# found from the build directory without a copy step.
target_compile_definitions(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# benchmark, not benchmark_main: this binary keeps its own main so DistributedContext::create
# sees argv before google-benchmark takes its share of it.
target_link_libraries(
    d2h2h2d_socket_d2h_bw_test
    PRIVATE
        test_metal_common_libs
        benchmark::benchmark
)

add_executable(d2h2h2d_socket_d2h_bw)
set_target_properties(
    d2h2h2d_socket_d2h_bw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(
    d2h2h2d_socket_d2h_bw
    PRIVATE
        d2h2h2d_socket_d2h_bw_test
        benchmark::benchmark
)

add_library(d2h2h2d_socket_h2d_bw_test OBJECT)
TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_h2d_bw_test)

target_sources(d2h2h2d_socket_h2d_bw_test PRIVATE test_h2d_bw.cpp)

target_include_directories(
    d2h2h2d_socket_h2d_bw_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
)

target_compile_definitions(
    d2h2h2d_socket_h2d_bw_test
    PRIVATE
        TT_METAL_HOST_BRIDGE=1
        TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(d2h2h2d_socket_h2d_bw_test PRIVATE test_metal_common_libs)

add_executable(d2h2h2d_socket_h2d_bw)
set_target_properties(
    d2h2h2d_socket_h2d_bw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(d2h2h2d_socket_h2d_bw PRIVATE d2h2h2d_socket_h2d_bw_test)

# Unlike the two device-leg tests this one is a real 2-rank MPI program, so it only builds
# when MPI is available.
if(TARGET OpenMPI::MPI)
    add_library(d2h2h2d_socket_h2h_bw_test OBJECT)
    TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_h2h_bw_test)

    target_sources(d2h2h2d_socket_h2h_bw_test PRIVATE test_h2h_bw.cpp)

    target_include_directories(
        d2h2h2d_socket_h2h_bw_test
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests
    )

    target_compile_definitions(d2h2h2d_socket_h2h_bw_test PRIVATE TT_METAL_HOST_BRIDGE=1)

    target_link_libraries(
        d2h2h2d_socket_h2h_bw_test
        PRIVATE
            test_metal_common_libs
            OpenMPI::MPI
    )

    add_executable(d2h2h2d_socket_h2h_bw)
    set_target_properties(
        d2h2h2d_socket_h2h_bw
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
    target_link_libraries(d2h2h2d_socket_h2h_bw PRIVATE d2h2h2d_socket_h2h_bw_test)

    add_library(d2h2h2d_socket_volume_d2d_test OBJECT)
    TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_volume_d2d_test)

    target_sources(d2h2h2d_socket_volume_d2d_test PRIVATE test_volume_d2d.cpp)

    target_include_directories(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests
    )

    target_compile_definitions(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            TT_METAL_HOST_BRIDGE=1
            TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            test_metal_common_libs
            OpenMPI::MPI
    )

    add_executable(d2h2h2d_socket_volume_d2d)
    set_target_properties(
        d2h2h2d_socket_volume_d2d
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
    target_link_libraries(d2h2h2d_socket_volume_d2d PRIVATE d2h2h2d_socket_volume_d2d_test)
endif()
