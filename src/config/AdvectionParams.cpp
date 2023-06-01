#include "AdvectionParams.h"

// ======================================================
// ======================================================
void ADVParams::setup(const ConfigMap& configMap)
{
  // geometry
  nx  = configMap.getInteger("geometry", "nx",  1024);
  nvx = configMap.getInteger("geometry", "nvx", 1024);
  n_fict_dim = configMap.getInteger("geometry", "n_fict_dim", 2048);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange3D");
  wg_size = configMap.getInteger("run", "workGroupSize", 512);

  // discretization parameters
  dt  = configMap.getFloat("discretization", "dt" , 0.0001);

  minRealx = configMap.getFloat("discretization", "minRealx", 0.0);
  maxRealx = configMap.getFloat("discretization", "maxRealx", 1.0);
  minRealVx= configMap.getFloat("discretization", "minRealVx", 0.0);
  maxRealVx= configMap.getFloat("discretization", "maxRealVx", 1.0);

  realWidthx = maxRealx - minRealx;
  realWidthVx = maxRealVx - minRealVx;
  dx = realWidthx / nx;
  dvx = (maxRealVx - minRealVx) / nvx;

  inv_dx     = 1/dx;
  inv_dvx     = 1/dvx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParams::print()
{
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  std::cout << "kernelImpl : " << kernelImpl << std::endl;
  std::cout << "wgSize     : " << wg_size << std::endl;
  printf( "gpu        : %d\n", gpu);
  printf( "maxIter    : %zu\n", maxIter);
  printf( "nx         : %zu\n", nx);
  printf( "nvx        : %zu\n", nvx);
  printf( "n_fict_dim : %zu\n", n_fict_dim);
  printf( "dt         : %f\n", dt);
  printf( "dx         : %f\n", dx);
  printf( "dvx        : %f\n", dvx);
  printf( "minRealx   : %f\n", minRealx );
  printf( "maxRealx   : %f\n", maxRealx);
  printf( "minRealVx  : %f\n", minRealVx);
  printf( "maxRealVx  : %f\n", maxRealVx);
  printf( "\n");

} // ADVParams::print
