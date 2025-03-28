#pragma once
#include "ConfigMap.h"
#include "types.hpp"

struct ADVParamsNonCopyable;

/**
 * Advection Parameters (declaration)
 */
struct ADVParams {
  ADVParams(ADVParamsNonCopyable &other);
  ADVParams(){update_deltas();};
  
  // Running on the GPU (false = CPU)
  bool gpu = false;

  //The percentage of n0 rows to compute in local memory
  float percent_loc = 1.0;

  // Outputs the solution to solution.log file to be read with the ipynb
  bool outputSolution = false;

  // Number of iterations
  size_t maxIter = 100;

  // Number of points for
  size_t n1 = 1024; //dimension of interest
  size_t n0 = 32;    //batch dimension (corresponds to velocities Vx)
  size_t n2 = 32;   //stride for x, is also a batch dimension
  //We get n2*n0 independent problems of size n1, and x has a stride of n2

  // Sizes of the SYCL work groups
  size_t pref_wg_size = 128;
  
  //Number of elements in dim0 and dim2 that a single work-item will process
  size_t seq_size0;
  size_t seq_size2;

  // Deltas : taille physique d'une cellule discrète (en x, vx, t)
  real_t dt  = 0.0001;
  real_t dx;
  real_t dvx;

  real_t inv_dx; // precompute inverse of dx

  // Min/max physical values of x (i.e., x[0] and x[-1])
  real_t minRealX   = 0;
  real_t maxRealX   = 1;
  real_t realWidthX = 1;

  // Min/max physical value of Vx
  real_t minRealVx = 0;
  real_t maxRealVx = 1;

  //! update physical values
  void update_deltas();
}; // struct ADVParams

//Need to have this in order to dodge the device trivially copyable SYCL
struct ADVParamsNonCopyable : ADVParams {
  ADVParamsNonCopyable(ADVParams &other);
  ADVParamsNonCopyable() = default;

  //The implementation of the kernel, correspond to core/impl cpp files
  std::string kernelImpl;

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();

};
