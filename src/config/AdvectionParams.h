#pragma once
#include "ConfigMap.h"
#include <unordered_map>
#include <iostream>

/**
 * Advection Parameters (declaration)
 */
struct ADVParams {
  // Running on the GPU (false = CPU)
  bool gpu;

  //The implementation of the kernel, correspond to core/impl cpp files
  std::string kernelImpl;

  // Number of iterations
  size_t maxIter;

  // Number of points for positions (x)
  size_t nx;

  // Number of points for speeds (Vx)
  size_t nVx;

  // Deltas : taille physique d'une cellule discrète (en x, vx, t)
  double dt;
  double dx;
  double dVx;

  double inv_dx; // precompute inverse of dx

  // Min/max physical values of x (i.e., x[0] and x[-1])
  double minRealx;
  double maxRealx;
  double realWidthx;

  // Min physical value of Vx
  double minRealVx;//we only need min since we perform advection in x only

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();
}; // struct ADVParams