#pragma once
#include "ConfigMap.h"
#include <unordered_map>
#include <iostream>

/**
 * Advection Parameters (declaration)
 */
struct ADVParams {
  // Running on the GPU (false = CPU)
  // bool gpu;

  //The implementation of the kernel, correspond to core/impl cpp files
  // std::string kernelImpl;

  // Outputs the solution to solution.log file to be read with the ipynb
  // bool outputSolution;

  // Number of iterations
  size_t maxIter;

  // Number of points for positions (x)
  size_t nx;

  // Number of points for speeds (Vx)
  size_t nvx;

  // Fictive dimension to get 1D independant problems with v and vx dimensions
  size_t n_fict_dim;

  // Sizes of the SYCL work groups
  size_t wg_size;

  // Deltas : taille physique d'une cellule discrète (en x, vx, t)
  double dt;
  double dx;
  double dvx;

  double inv_dx; // precompute inverse of dx
  double inv_dvx;

  // Min/max physical values of x (i.e., x[0] and x[-1])
  double minRealx;
  double maxRealx;
  double realWidthx;

  // Min/max physical value of Vx
  double minRealVx;
  double maxRealVx;
  double realWidthVx;

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();
}; // struct ADVParams