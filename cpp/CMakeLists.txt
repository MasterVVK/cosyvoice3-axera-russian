# CosyVoice3 AXERA C++ Runtime (AXCL Host Mode)
# Runs on CM3588 host, accesses AX650N NPU via PCIe
# No MNN dependency - CosyVoice3 runs HiFT Part1 on NPU (axmodel)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

cmake_minimum_required(VERSION 3.13)
project(COSYVOICE3-AXCL-CPP)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Source directories ---
# ax-llm (axcl-context branch): AXCL model runner + utils
set(AXLLM_DIR ${CMAKE_SOURCE_DIR}/../../ax-llm-axcl)
# CosyVoice2.Axera: shared utilities (files, slice_3d, concat_3d, etc.)
set(CV2_SRC_DIR ${CMAKE_SOURCE_DIR}/../../Cosyvoice2.Axera/cpp)

message(STATUS "AXLLM_DIR = ${AXLLM_DIR}")
message(STATUS "CV2_SRC_DIR = ${CV2_SRC_DIR}")

# --- AXCL SDK ---
# On CM3588 with axclhost package: headers in /usr/include/axcl/, libs in /usr/lib/axcl/
if(NOT AXCL_INCLUDE_DIR)
    if(EXISTS /usr/include/axcl/axcl.h)
        set(AXCL_INCLUDE_DIR /usr/include/axcl)
    elseif(EXISTS ${AXLLM_DIR}/third_party/axcl/include/axcl.h)
        set(AXCL_INCLUDE_DIR ${AXLLM_DIR}/third_party/axcl/include)
    else()
        message(FATAL_ERROR "AXCL SDK not found. Install axclhost package or set AXCL_INCLUDE_DIR.")
    endif()
endif()

if(NOT AXCL_LIB_DIR)
    if(EXISTS /usr/lib/axcl/libaxcl_rt.so)
        set(AXCL_LIB_DIR /usr/lib/axcl)
    elseif(EXISTS /usr/lib/libaxcl_rt.so)
        set(AXCL_LIB_DIR /usr/lib)
    else()
        set(AXCL_LIB_DIR /usr/lib/axcl)
    endif()
endif()

message(STATUS "AXCL_INCLUDE_DIR = ${AXCL_INCLUDE_DIR}")
message(STATUS "AXCL_LIB_DIR = ${AXCL_LIB_DIR}")

include_directories(${AXCL_INCLUDE_DIR})
link_directories(${AXCL_LIB_DIR})

# --- Include paths ---
# Our sources first (includes CV2_Tokenizer.hpp wrapper to avoid ax-llm Tokenizer conflict)
include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)
# ax-llm-axcl: AXCL model runner (with dev_id), AXCL manager, bfloat16, LLMEmbedSelector
include_directories(${AXLLM_DIR}/src)
include_directories(${AXLLM_DIR}/src/runner)
include_directories(${AXLLM_DIR}/src/runner/utils)
include_directories(${AXLLM_DIR}/src/runner/ax_model_runner)
# CV2: utility files (files.cpp, mrope.cpp, slice_3d, wav, utils.hpp, etc.)
# CV2 comes AFTER ax-llm: ax-llm wins for ax_model_runner, Tokenizer uses CV2_Tokenizer.hpp wrapper
include_directories(${CV2_SRC_DIR}/src)
include_directories(${CV2_SRC_DIR}/src/runner)
include_directories(${CV2_SRC_DIR}/src/runner/utils)

# --- OpenCV ---
if(NOT OpenCV_DIR)
    if(EXISTS ${CV2_SRC_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
        set(OpenCV_DIR ${CV2_SRC_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
    endif()
endif()
message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- Build ---
function(build_exec name main_source)
    add_executable(${name} ${main_source}
                    # AXCL model runner (replaces BSP ax_engine runner)
                    ${AXLLM_DIR}/src/runner/ax_model_runner/ax_model_runner_ax650.cpp
                    # AXCL manager (worker thread for PCIe)
                    ${AXLLM_DIR}/src/runner/utils/axcl_manager.cpp
                    # Shared utilities from ax-llm
                    ${AXLLM_DIR}/src/runner/utils/memory_utils.cpp
                    ${AXLLM_DIR}/src/runner/utils/cqdm.cpp
                    # Use CV2 Tokenizer (has Init(path,bos,eos) + Encode with ImageInfo)
                    ${CV2_SRC_DIR}/src/runner/Tokenizer/Tokenizer.cpp
                    # Shared utilities from CV2 (not in ax-llm)
                    ${CV2_SRC_DIR}/src/runner/utils/files.cpp
                    ${CV2_SRC_DIR}/src/runner/utils/mrope.cpp
                    ${CV2_SRC_DIR}/src/runner/utils/slice_3d.tpp
                    )

    # AXCL runtime + sys (for AXCL_SYS_MinvalidateCache)
    target_link_libraries(${name} axcl_rt axcl_sys)
    target_link_libraries(${name} ${OpenCV_LIBS})
    target_link_libraries(${name} pthread)
    install(TARGETS ${name} DESTINATION bin)
endfunction()

build_exec(cosyvoice3_tts src/main.cpp)

file(GLOB RUN_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/*.py" "${CMAKE_SOURCE_DIR}/scripts/*.sh")
install(FILES ${RUN_SCRIPT} DESTINATION bin/)
