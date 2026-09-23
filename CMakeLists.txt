cmake_minimum_required(VERSION 3.15)
project(ad-benchmarks
    LANGUAGES CXX
    DESCRIPTION "Comparative benchmark suite for C++ automatic differentiation libraries (XAD, CppAD, Adept 2, autodiff) on quantitative-finance workloads."
    HOMEPAGE_URL "https://github.com/auto-differentiation/ad-benchmarks"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ===========================================================================
# Options - enable/disable each library
# ===========================================================================
option(ENABLE_XAD      "Build XAD benchmarks"      ON)
option(ENABLE_CPPAD    "Build CppAD benchmarks"     ON)
option(ENABLE_ADEPT    "Build Adept 2 benchmarks"   ON)
option(ENABLE_AUTODIFF "Build autodiff benchmarks"  ON)
option(ENABLE_XAD_JIT  "Build XAD JIT benchmarks"   OFF)  # requires xad-codegen

set(MC_PATHS 10000 CACHE STRING "Default MC path count for LIBOR swaption benchmark")

# Path to local XAD if not system-installed
set(XAD_DIR "" CACHE PATH "Path to XAD source (if not using FetchContent)")
set(XAD_CODEGEN_DIR "" CACHE PATH "Path to xad-codegen source (for JIT benchmarks)")

# ===========================================================================
# Compiler flags for fair comparison
# ===========================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -DNDEBUG -mavx2 -mfma)
elseif(MSVC)
    # /fp:fast lets MSVC contract math ops (FMA), drop errno from <cmath> calls,
    # and reorder FP — closer parity with GCC -O3 default math behavior.
    add_compile_options(/O2 /DNDEBUG /arch:AVX2 /fp:fast)
endif()

# ===========================================================================
# Fetch dependencies
# ===========================================================================

# --- XAD ---
if(ENABLE_XAD)
    set(XAD_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    if(ENABLE_XAD_JIT)
        # XAD handles xad-codegen integration internally via XAD_CODEGEN_DIR
        set(XAD_ENABLE_CODEGEN ON CACHE BOOL "" FORCE)
        if(NOT XAD_CODEGEN_DIR)
            message(FATAL_ERROR "ENABLE_XAD_JIT requires XAD_CODEGEN_DIR to be set")
        endif()
    endif()
    if(XAD_DIR)
        add_subdirectory(${XAD_DIR} ${CMAKE_BINARY_DIR}/xad)
    else()
        FetchContent_Declare(xad
            GIT_REPOSITORY https://github.com/auto-differentiation/xad.git
            GIT_TAG        main
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(xad)
    endif()
endif()

# --- CppAD ---
if(ENABLE_CPPAD)
    FetchContent_Declare(cppad
        GIT_REPOSITORY https://github.com/coin-or/CppAD.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(cppad)
endif()

# --- Adept 2 ---
if(ENABLE_ADEPT)
    FetchContent_Declare(adept
        GIT_REPOSITORY https://github.com/rjhogan/Adept-2.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_Populate(adept)
    # Adept doesn't have a proper CMake build. Build manually.
    add_library(adept STATIC
        ${adept_SOURCE_DIR}/adept/Stack.cpp
        ${adept_SOURCE_DIR}/adept/StackStorageOrig.cpp
        ${adept_SOURCE_DIR}/adept/Storage.cpp
    )
    target_include_directories(adept PUBLIC ${adept_SOURCE_DIR}/include)
    target_compile_definitions(adept PUBLIC ADEPT_STACK_THREAD_UNSAFE)
    # Adept needs config.h — generate a minimal one
    file(WRITE ${adept_BINARY_DIR}/config.h
        "#ifndef ADEPT_CONFIG_H\n#define ADEPT_CONFIG_H\n#define ADEPT_VERSION \"2.1.1\"\n#endif\n")
    target_include_directories(adept PRIVATE ${adept_BINARY_DIR})
endif()

# --- autodiff ---
if(ENABLE_AUTODIFF)
    FetchContent_Declare(autodiff
        GIT_REPOSITORY https://github.com/autodiff/autodiff.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
    )
    set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(autodiff)
endif()

# ===========================================================================
# Benchmark executable
# ===========================================================================
set(BENCH_SOURCES src/main.cpp src/fd_benchmarks.cpp)

if(ENABLE_XAD)
    list(APPEND BENCH_SOURCES
        xad/heston_xad.cpp
        xad/sabr_calibration_xad.cpp
        xad/xva_xad.cpp
        xad/libor_swaption_xad.cpp
    )
endif()

if(ENABLE_XAD_JIT)
    list(APPEND BENCH_SOURCES
        xad/heston_jit.cpp
        xad/sabr_calibration_jit.cpp
        xad/xva_jit.cpp
        xad/libor_swaption_jit.cpp
    )
endif()

if(ENABLE_CPPAD)
    list(APPEND BENCH_SOURCES
        cppad/heston_cppad.cpp
        cppad/sabr_calibration_cppad.cpp
        cppad/xva_cppad.cpp
        cppad/libor_swaption_cppad.cpp
    )
endif()

if(ENABLE_ADEPT)
    list(APPEND BENCH_SOURCES
        adept/heston_adept.cpp
        adept/sabr_calibration_adept.cpp
        adept/xva_adept.cpp
        adept/libor_swaption_adept.cpp
    )
endif()

if(ENABLE_AUTODIFF)
    list(APPEND BENCH_SOURCES
        autodiff/heston_autodiff.cpp
        autodiff/sabr_calibration_autodiff.cpp
        autodiff/xva_autodiff.cpp
        autodiff/libor_swaption_autodiff.cpp
    )
endif()

add_executable(ad_benchmarks ${BENCH_SOURCES})
target_include_directories(ad_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compile definitions for enabled libraries
if(ENABLE_XAD)
    target_compile_definitions(ad_benchmarks PRIVATE ENABLE_XAD)
    target_link_libraries(ad_benchmarks PRIVATE xad)
endif()

if(ENABLE_CPPAD)
    target_compile_definitions(ad_benchmarks PRIVATE ENABLE_CPPAD)
    target_include_directories(ad_benchmarks PRIVATE ${cppad_SOURCE_DIR}/include)
endif()

if(ENABLE_ADEPT)
    target_compile_definitions(ad_benchmarks PRIVATE ENABLE_ADEPT)
    target_link_libraries(ad_benchmarks PRIVATE adept)
endif()

if(ENABLE_AUTODIFF)
    target_compile_definitions(ad_benchmarks PRIVATE ENABLE_AUTODIFF)
    target_link_libraries(ad_benchmarks PRIVATE autodiff::autodiff)
endif()

if(ENABLE_XAD_JIT)
    target_compile_definitions(ad_benchmarks PRIVATE ENABLE_XAD_JIT)
    target_link_libraries(ad_benchmarks PRIVATE xad_codegen)
    target_include_directories(ad_benchmarks PRIVATE ${XAD_CODEGEN_DIR})
endif()
