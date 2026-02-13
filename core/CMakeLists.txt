# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 14)

if(NOT target)
  set(target instantvnr)
endif()
message(STATUS "target - ${target}")

include(configure_cuda) # skip optix here

if(DEFINED MACROCELL_SIZE_MIP)
  add_definitions(-DMACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP})
else()
  add_definitions(-DMACROCELL_SIZE_MIP=4)
endif()

if(ENABLE_FVSRN)
  add_definitions(-DKERNEL_DOUBLE_PRECISION=1) # host: double precision for unit tests
  set(DISABLE_ADAPTIVE_SAMPLING ON)
endif()

if(DISABLE_ADAPTIVE_SAMPLING)
  add_compile_definitions(ADAPTIVE_SAMPLING=0)
else()
  add_compile_definitions(ADAPTIVE_SAMPLING=1)
endif()

# ======================================================================== #
# Add the diff-gaussian-rasterization subdirectory
# ======================================================================== #
set(DIFF_GAUSSIAN_RASTERIZATION_DIR ${CMAKE_CURRENT_LIST_DIR}/../../diff-gaussian-rasterization)
set(HAS_DIFF_GAUSSIAN_RASTERIZATION FALSE)
if(EXISTS ${DIFF_GAUSSIAN_RASTERIZATION_DIR}/CMakeLists.txt)
  set(HAS_DIFF_GAUSSIAN_RASTERIZATION TRUE)
endif()

# This makes the CudaRasterizer target available and handles all dependencies.
# Skip it when diff-gaussian-rasterization is not checked out.
if(HAS_DIFF_GAUSSIAN_RASTERIZATION)
  if(NOT TARGET CudaRasterizer)
    add_subdirectory(${DIFF_GAUSSIAN_RASTERIZATION_DIR} ${CMAKE_CURRENT_BINARY_DIR}/diff-gaussian-rasterization)
  endif()
  set_target_properties(CudaRasterizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  message(STATUS "diff-gaussian-rasterization not found; skipping CudaRasterizer target.")
endif()

# ======================================================================== #
# renderer core
# ======================================================================== #

add_library(${target} SHARED
  renderer/method_pathtracing.cu
  renderer/method_pathtracing.h
  renderer/method_raymarching.cu
  renderer/method_raymarching.h
  renderer/raytracing.h
  renderer/dda.h

  samplers/neural_sampler.cu
  samplers/neural_sampler.cpp
  samplers/neural_sampler.h
  sampler.cu
  sampler.h

  network.cu
  network.h

  macrocell.cu
  macrocell.h

  instantvnr_types.cu
  instantvnr_types.h

  array.h

  serializer.cpp
  serializer.h

  renderer.cpp
  renderer.h

  ../api.cpp
  ../api.h
)

if(HAS_DIFF_GAUSSIAN_RASTERIZATION)
  target_include_directories(${target} PRIVATE
    ../../diff-gaussian-rasterization
    ../../diff-gaussian-rasterization/cuda_rasterizer
    ../../diff-gaussian-rasterization/third_party/cuBQL
    ../../diff-gaussian-rasterization/third_party/glm
  )
endif()

if(ENABLE_IN_SHADER)
  target_sources(${target} PRIVATE
    networks/tcnn_device_encoder.cu
    networks/tcnn_device_network.cu
    marching_cube.cu
  )
  target_compile_definitions(${target} PUBLIC ENABLE_IN_SHADER=1)
endif()

if(ENABLE_OUT_OF_CORE)
  target_compile_definitions(${target} PRIVATE ENABLE_OUT_OF_CORE=1)
  target_link_libraries     (${target} PRIVATE $<$<PLATFORM_ID:Linux>:aio>)
endif()

if(ENABLE_FVSRN AND HAS_DIFF_GAUSSIAN_RASTERIZATION)
  target_sources(${target} PRIVATE
    ../../diff-gaussian-rasterization/gaussian_model.h
    networks/fvsrn_network.cu
  )
  target_compile_definitions(${target} PUBLIC ENABLE_FVSRN=1)
elseif(ENABLE_FVSRN)
  message(STATUS "ENABLE_FVSRN requested but diff-gaussian-rasterization is missing; disabling FVSRN sources.")
endif()

set_target_properties(${target} 
PROPERTIES 
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>)
target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>)

target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">)   # -> variable declared but not referenced
target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044">) # -> extern variable '...' treated as static

target_link_libraries(${target} PRIVATE TBB::tbb curand)

# Link to the CudaRasterizer target which provides all the necessary dependencies
target_link_libraries(${target} PUBLIC # intentionally keep these library public
  gdt util tfnmodule tiny-cuda-nn
)

if(TARGET CudaRasterizer)
  target_link_libraries(${target} PUBLIC CudaRasterizer) # intentionally keep this library public
endif()

if(TARGET openvkl::openvkl AND TARGET openvkl::testing)
  target_compile_definitions(${target} PRIVATE ENABLE_OPENVKL=1)
  target_link_libraries(${target} PRIVATE openvkl::openvkl openvkl::testing)
endif()

if(ENABLE_LOGGING)
  target_compile_definitions(${target} PRIVATE ENABLE_LOGGING=1)
endif()

# target_include_directories(vnr_core PUBLIC $<$<PLATFORM_ID:Windows>:${INTEL_AIO_INC}>)
# target_link_libraries     (vnr_core PUBLIC $<$<PLATFORM_ID:Windows>:${INTEL_AIO_LIB}>)
# target_compile_definitions(vnr_core PUBLIC AIO_INTEL)
# target_compile_definitions(vnr_core PRIVATE TCNN_NAMESPACE=tcnn)
