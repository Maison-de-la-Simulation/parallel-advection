#pragma once
#include "ConfigMap.h"
#include <iostream>

/**
 * Simulation run parameters, which we don't need in advection kernels
 */
struct InitParams {
  // Running on the GPU (false = CPU)
  bool gpu;

  //The implementation of the kernel, correspond to core/impl cpp files
  std::string kernelImpl;

  // Outputs the solution to solution.log file to be read with the ipynb
  bool outputSolution;

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();
  void print_kokkos();
}; // struct InitParams