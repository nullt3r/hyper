// CUDA_CHECK: a CUDA call's error as an exception with the file and line
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)
