#include "Adv4dParams.hpp"
#include <iostream>

// Load values from ConfigMap and compute derived quantities
void Adv4dParams::setup(const ConfigMap& configMap) {
  // Grid sizes
  nx  = configMap.getInteger("problem", "nx", 128);
  nvx = configMap.getInteger("problem", "nvx", 128);
  ny  = configMap.getInteger("problem", "ny", 128);
  nvy = configMap.getInteger("problem", "nvy", 128);

  // Time step
  dt = configMap.getFloat("problem", "dt", 0.001);

  // Physical domain
  minX  = configMap.getFloat("problem", "minX", 0.0);
  maxX  = configMap.getFloat("problem", "maxX", 1.0);
  minVx = configMap.getFloat("problem", "minVx", -1.0);
  maxVx = configMap.getFloat("problem", "maxVx",  1.0);
  minY  = configMap.getFloat("problem", "minY",  0.0);
  maxY  = configMap.getFloat("problem", "maxY",  1.0);
  minVy = configMap.getFloat("problem", "minVy", -1.0);
  maxVy = configMap.getFloat("problem", "maxVy",  1.0);

  // Optimization
  gpu = configMap.getBool("optimization", "gpu", true);
  pref_wg_size = configMap.getInteger("optimization", "pref_wg_size", 128);

  // Compute deltas and inverses
  update_deltas();
}

// Compute cell sizes and their inverses
void Adv4dParams::update_deltas() {
  dx  = (maxX  - minX)  / nx;
  dvx = (maxVx - minVx) / nvx;
  dy  = (maxY  - minY)  / ny;
  dvy = (maxVy - minVy) / nvy;

  inv_dx  = 1.0 / dx;
  inv_dvx = 1.0 / dvx;
  inv_dy  = 1.0 / dy;
  inv_dvy = 1.0 / dvy;
}

// Print parameters to stdout
void Adv4dParams::print() const {
  std::cout << "############ Adv4dParams ############" << std::endl;
  std::cout << "gpu           : " << gpu << std::endl;
  std::cout << "pref_wg_size  : " << pref_wg_size << std::endl;
  std::cout << "dt            : " << dt << std::endl;
  std::cout << "nx, nvx       : " << nx << ", " << nvx << std::endl;
  std::cout << "ny, nvy       : " << ny << ", " << nvy << std::endl;
  std::cout << "minX, maxX    : " << minX << ", " << maxX << std::endl;
  std::cout << "minVx, maxVx  : " << minVx << ", " << maxVx << std::endl;
  std::cout << "minY, maxY    : " << minY << ", " << maxY << std::endl;
  std::cout << "minVy, maxVy  : " << minVy << ", " << maxVy << std::endl;
  std::cout << "dx, dvx       : " << dx  << ", " << dvx << std::endl;
  std::cout << "dy, dvy       : " << dy  << ", " << dvy << std::endl;
  std::cout << "inv_dx        : " << inv_dx  << std::endl;
  std::cout << "inv_dvx       : " << inv_dvx << std::endl;
  std::cout << "inv_dy        : " << inv_dy  << std::endl;
  std::cout << "inv_dvy       : " << inv_dvy << std::endl;
  std::cout << "#####################################" << std::endl;
}
