#pragma once
#include "ConfigMap.h"

struct ADVParamsNonCopyable;

/**
 * Advection Parameters (declaration)
 */
struct ADVParams {
  ADVParams(ADVParamsNonCopyable &other);
  ADVParams() = default;
  
  // Running on the GPU (false = CPU)
  bool gpu = false;

  //The implementation of the kernel, correspond to core/impl cpp files
  // std::string kernelImpl;

  // Outputs the solution to solution.log file to be read with the ipynb
  bool outputSolution = false;

  // Number of iterations
  size_t maxIter = 100;

  // Number of points for positions (x) velocities (vx) and fictive dim (z)
  size_t nx  = 1024;
  size_t nvx = 32;
  size_t nz  = 32;

  // Sizes of the SYCL work groups
  size_t wg_size = 128;

  // Deltas : taille physique d'une cellule discrète (en x, vx, t)
  double dt  = 0.0001;
  double dx;
  double dvx;

  double inv_dx; // precompute inverse of dx

  // Min/max physical values of x (i.e., x[0] and x[-1])
  double minRealX   = 0;
  double maxRealX   = 1;
  double realWidthX = 1;

  // Min/max physical value of Vx
  double minRealVx = 0;
  double maxRealVx = 1;

  //! update physical values
  void update_deltas();
}; // struct ADVParams

//Need to have this in order to dodge the device trivially copyable SYCL
struct ADVParamsNonCopyable : ADVParams {
  //The implementation of the kernel, correspond to core/impl cpp files
  std::string kernelImpl;

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();

};


// #ifdef SYCL_DEVICE_COPYABLE 
// template<>
// struct sycl::is_device_copyable<ADVParams> : std::true_type {};
// #endif
