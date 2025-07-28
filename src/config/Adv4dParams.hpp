#pragma once
#include "ConfigMap.h"
#include <types.hpp>

/**
 * 4D Advection Parameters
 */
struct Adv4dParams {
  // Constructors
  Adv4dParams() = default;

  // Kernel execution target
  bool gpu = false;

  // Work-group optimization
  size_t pref_wg_size = 128;

  // Grid sizes
  size_t nx  = 128;
  size_t nvx = 128;
  size_t ny  = 128;
  size_t nvy = 128;

  size_t n0 = -1;
  size_t n1 = -1;
  size_t n2 = -1;

  // Time step
  real_t dt = 0.001;

  // Physical bounds
  real_t minX  = 0.0;
  real_t maxX  = 1.0;
  real_t minVx = -1.0;
  real_t maxVx = 1.0;
  real_t minY  = 0.0;
  real_t maxY  = 1.0;
  real_t minVy = -1.0;
  real_t maxVy = 1.0;

  // Cell sizes
  real_t dx, dvx, dy, dvy;

  // Inverse cell sizes
  real_t inv_dx, inv_dvx, inv_dy, inv_dvy;

  // Initialization
  void setup(const ConfigMap& configMap);

  // Precompute deltas and inverses
  void update_deltas();

  // Logging
  void print() const;
};
