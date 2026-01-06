#include <string>
#include <vector>
#include <random>
#include <fstream>

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <mma.h>

#include "fvsrn_network.h"

// defined in the model (should use JIT instead)
#include "../../../diff-gaussian-rasterization/gaussian_model.h"

namespace vnr {

struct FvsrnNetwork::Impl {
	GaussianModel model;
	CUDABuffer constant; 
};

FvsrnNetwork::FvsrnNetwork() : pimpl(new Impl()) {}

FvsrnNetwork::~FvsrnNetwork() { pimpl->model.cleanup(); pimpl.reset(); }

int FvsrnNetwork::n_neurons() const { return pimpl->model.getNumGaussians(); }

void* FvsrnNetwork::network_direct_access() { return 0; }

void FvsrnNetwork::deserialize_params(const json& config) {
    std::cout<<"params"<<std::endl;
	std::string plyfile = config["ply"];

	bool success = pimpl->model.loadPly(plyfile);

	assert(success);
}

void FvsrnNetwork::deserialize_model(json config) {
  std::cout<<"model"<<std::endl;
	std::string plyfile = config["ply"];

	bool success = pimpl->model.loadPly(plyfile);

	assert(success);
}

void FvsrnNetwork::infer(const GPUMatrixDynamic<float>& coord, GPUMatrixDynamic<float>& output, cudaStream_t stream) const {
  TRACE_CUDA;

  assert(coord.layout() == TCNN_NAMESPACE::MatrixLayout::ColumnMajor && "input coordinate should be a column major matrix");
  assert(coord.m() == 3 && "incorrect coordinate buffer shape");
  assert(output.m() == 1 && "incorrect coordinate buffer shape");
  int num_samples = coord.n();
  
  // Synchronize the stream before any GPU operations (since infer doesn't support streams)
  float* d_out_weights;
  cudaMalloc((void**)&d_out_weights, num_samples * sizeof(float));
  
  // Call the modified infer function with correctly formatted data
  bool success = pimpl->model.infer(
    coord.data(),    // GPU buffer with coordinates
    num_samples,             // number of sample points
    output.data(),           // GPU buffer for output values (writes directly to output matrix)
    d_out_weights,           // GPU buffer for output weights (temporary, required by API)
    1.0f,                    // scale_modifier (default)
    0.0f,                    // background (default)
    false,                   // use_gaussian_bvh (default)
    false                    // debug (default)
  );
  
  cudaFree(d_out_weights);
  
  if (!success) {
    throw std::runtime_error("Inference failed in FvsrnNetwork::infer");
  }
  cudaDeviceSynchronize();
  
  TRACE_CUDA;
}

}
