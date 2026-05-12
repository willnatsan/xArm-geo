cmake_minimum_required(VERSION 3.25)
project(xArm_geo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle these to enable/disable specific backends
option(BUILD_WITH_MUJOCO "Build with MuJoCo simulation support" ON)
option(BUILD_WITH_REAL_XARM "Build with physical xArm SDK support" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# --- Core Dependencies ---
include(FetchContent)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_Declare(
  smooth
  GIT_REPOSITORY https://github.com/pettni/smooth.git
  GIT_TAG 08c5d55
)
FetchContent_MakeAvailable(smooth)

find_package(yaml-cpp REQUIRED)
find_package(proxsuite REQUIRED)
find_package(coal REQUIRED)
find_package(tinyxml2 REQUIRED)

# --- Optional Dependencies ---

if(BUILD_PYTHON_BINDINGS)
  message(STATUS "xArm_geo: Enabling Python Bindings (nanobind)")
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE nanobind_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_DIR}" NO_DEFAULT_PATH)
endif()

if(BUILD_WITH_MUJOCO)
    message(STATUS "xArm_geo: Enabling MuJoCo Backend")
    find_package(mujoco REQUIRED)
    find_package(glfw3 REQUIRED)
    add_compile_definitions(XARM_GEO_HAS_MUJOCO)
endif()

# xArm SDK (Hardware Interface; IF Enabled)
# Using FetchContent, as it is not packaged on conda-forge.
if(BUILD_WITH_REAL_XARM)
    message(STATUS "xArm_geo: Enabling Real xArm Backend")
    FetchContent_Declare(
            xarm_sdk
            GIT_REPOSITORY https://github.com/xArm-Developer/xArm-CPLUS-SDK.git
            GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(xarm_sdk)

    # Manually build xArm lib from sources
    file(GLOB_RECURSE ALL_SDK_SOURCES "${xarm_sdk_SOURCE_DIR}/src/*")
    set(XARM_SDK_SOURCES "")
    foreach(file ${ALL_SDK_SOURCES})
        string(FIND "${file}" "/bak/" IS_BAK_FILE)
        if(IS_BAK_FILE EQUAL -1)
            list(APPEND XARM_SDK_SOURCES ${file})
        endif()
    endforeach()

    add_library(xarm STATIC ${XARM_SDK_SOURCES})
    target_include_directories(xarm PUBLIC "${xarm_sdk_SOURCE_DIR}/include")
    target_compile_options(xarm PRIVATE -w) # Suppressing Warnings from xArm SDK
    target_link_libraries(xarm PRIVATE Threads::Threads)
    add_compile_definitions(XARM_GEO_HAS_REAL_XARM)
endif()

# --- Asset (URDF & MuJoCo XML) Generation ---

set(XARM_DESCRIPTION_PATH "${PROJECT_SOURCE_DIR}/assets")
configure_file(
        ${PROJECT_SOURCE_DIR}/config/config.h.in
        ${PROJECT_BINARY_DIR}/xarm_geo_config.h
)

set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)
set(URDF_DIR ${ASSETS_DIR}/urdf)
set(MJCF_DIR ${ASSETS_DIR}/mjcf)
set(GENERATOR_SCRIPT ${ASSETS_DIR}/gen_description_files.py)
set(GENERATOR_TARGET_NAME "gen_description_files")
set(XACRO_SOURCE_FILE ${URDF_DIR}/xarm_device.urdf.xacro)

set(GENERATED_FILES
        ${URDF_DIR}/_compiled/xarm5.urdf
        ${URDF_DIR}/_compiled/xarm6.urdf
        ${URDF_DIR}/_compiled/xarm7.urdf
        ${URDF_DIR}/_compiled/lite6.urdf
        ${URDF_DIR}/_compiled/uf850.urdf
        ${URDF_DIR}/_compiled/xarm7_mirror.urdf
        ${MJCF_DIR}/xarm5.xml
        ${MJCF_DIR}/xarm6.xml
        ${MJCF_DIR}/xarm7.xml
        ${MJCF_DIR}/lite6.xml
        ${MJCF_DIR}/uf850.xml
        ${MJCF_DIR}/xarm7_mirror.xml
)

add_custom_command(
        OUTPUT ${GENERATED_FILES}
        COMMAND ${Python_EXECUTABLE} ${GENERATOR_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${GENERATOR_SCRIPT} ${XACRO_SOURCE_FILE}
        COMMENT "Generating Description Files"
)
add_custom_target(${GENERATOR_TARGET_NAME} ALL DEPENDS ${GENERATED_FILES})

# --- Building Library ---

set(LIB_SOURCES
        src/xarm_geo.cpp
        src/trajectory/trajectory.cpp
        src/core/system.cpp
        src/interfaces/hardware.cpp
        src/interfaces/simulation.cpp
        src/modelling/collision.cpp
        src/modelling/dynamics.cpp
        src/modelling/kinematics.cpp
        src/modelling/optimal_kinematics.cpp
        src/safety/asif.cpp
        src/safety/barriers.cpp
        src/safety/constraints.cpp
        src/safety/tasks.cpp
        src/control/admittance.cpp
        src/control/controller.cpp
        src/control/feedback.cpp
        src/utils/model_builder.cpp
        src/utils/model_config.cpp
        src/utils/parsing_utils.cpp
)

if(BUILD_WITH_MUJOCO)
    # list(APPEND LIB_SOURCES src/interface/sim.cpp)
endif()

if(BUILD_WITH_REAL_XARM)
    # list(APPEND LIB_SOURCES src/interface/hardware.cpp)
endif()

add_library(${PROJECT_NAME} ${LIB_SOURCES})

# --- Include Directories & Linking ---

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

# Core Links
target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC Eigen3::Eigen smooth proxsuite::proxsuite coal::coal
        PRIVATE yaml-cpp::yaml-cpp tinyxml2::tinyxml2
)

# Optional Links
if(BUILD_WITH_MUJOCO)
    target_link_libraries(${PROJECT_NAME} PUBLIC mujoco::mujoco glfw)
endif()

if(BUILD_WITH_REAL_XARM)
    target_link_libraries(${PROJECT_NAME} PUBLIC xarm)
endif()

add_dependencies(${PROJECT_NAME} ${GENERATOR_TARGET_NAME})

# --- Executables / Tests ---

if(BUILD_WITH_MUJOCO)
    add_executable(simulation tests/simulation_test.cpp)
    target_link_libraries(simulation PRIVATE ${PROJECT_NAME})
endif()

if (BUILD_WITH_REAL_XARM)
    add_executable(hardware tests/hardware_test.cpp)
    target_link_libraries(hardware PRIVATE ${PROJECT_NAME})
endif()

add_executable(examples_compilation tests/examples_compile_test.cpp)
target_link_libraries(examples_compilation PRIVATE ${PROJECT_NAME})
