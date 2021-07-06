cmake_minimum_required(VERSION 3.5)
project(5point_alg)

# Compiler Options -----------------------------------------------------------------------------------------------------
SET(CMAKE_CONFIGURATION_TYPES RELEASE; DEBUG)
SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build Type: Release" FORCE)

if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
            OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    #list(GET GCC_VERSION_COMPO5TS 1 GCC_MINOR)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    if (GCC_VERSION VERSION_GREATER 4.7)
        add_definitions(-std=c++11)
    elseif(GCC_VERSION VERSION_GREATER 5.1)
        add_definitions(-std=c++14)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # extend release-profile with fast-math
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math -march=native")
endif()
set(PYBIND11_CPP_STANDARD -std=c++11)

# Third-party libs -----------------------------------------------------------------------------------------------------

INCLUDE_DIRECTORIES(
        ./third_party_libs/
        ./third_party_libs/include
        ./third_party_libs/zlib/include
        ./third_party_libs/ann_1.1_char/include
        ./third_party_libs/matrix
        ./third_party_libs/zlib
        ./third_party_libs/5point
        ./third_party_libs/getopt
)

add_subdirectory(./third_party_libs)

set (THIRD_PARTY_LIBS
        5point
        )

## OpenMP
#find_package(OpenMP REQUIRED)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

# Python Interp Options ------------------------------------------------------------------------------------------------
# set(PYTHON_EXECUTABLE /Users/corsy/anaconda3/bin/python)
# set(PYTHON_LIBRARY /Users/corsy/anaconda3/lib/libpython3.6m.dylib)
set(PYBIND11_PYTHON_VERSION 3.7)

# Soft inlier PnP  -----------------------------------------------------------------------------------------------------
find_package(pybind11 REQUIRED)
pybind11_add_module(5point_alg
        5point_alg_interface.cpp
        )
target_link_libraries(5point_alg PUBLIC ${THIRD_PARTY_LIBS})
