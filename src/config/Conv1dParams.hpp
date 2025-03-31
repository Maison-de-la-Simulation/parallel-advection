#pragma once
#include "ConfigMap.h"
#include <types.hpp>

struct Conv1dParamsNonCopyable;

/**
 * Advection Parameters (declaration)
 */
struct Conv1dParams {
  Conv1dParams(Conv1dParamsNonCopyable &other);
  Conv1dParams(){};
  
  size_t length = 1024;
  short unsigned channel_in = 3;
  short unsigned channel_out;// = channel_in # constraint
  short unsigned k = 3;
  size_t total_batch_size = 262144; //512*512
  size_t batch_size_n2 = 512;

  size_t n0;
  size_t n1;
  size_t n2;
  size_t n_write;

  bool gpu = true;
  short unsigned pref_wg_size = 512;
  size_t seq_size0 = 1;
  size_t seq_size2 = 1;
  bool inplace = true;

  size_t compute_output_size(size_t Lin, short unsigned kernel_size);
}; // struct Conv1dParams

//Need to have this in order to dodge the device trivially copyable SYCL
struct Conv1dParamsNonCopyable : Conv1dParams {
  Conv1dParamsNonCopyable(Conv1dParams &other);
  Conv1dParamsNonCopyable() = default;

  std::string kernelImpl;

  void setup(const ConfigMap& configMap); 
  void print();
};
