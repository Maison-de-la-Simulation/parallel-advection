#include "AdvectionParams.h"

// ======================================================
// ======================================================
void ADVParams::setup(const ConfigMap& configMap)
{
  // geometry
  nx  = configMap.getInteger("geometry", "nx",  512);
  nVx = configMap.getInteger("geometry", "nVx", 4);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", nx/2);
  gpu = configMap.getBool("run", "gpu", false);

  strKernelImpl = configMap.getString("run", "kernelImpl", "BasicRange");

  if (enumMap.find(strKernelImpl) != enumMap.end()) {
      kernelImpl = enumMap[strKernelImpl];
  } else {
      throw std::runtime_error("Invalid enum value name");
  }

  // discretization parameters
  dt  = configMap.getFloat("discretization", "dt" , 1.0);
  dx  = configMap.getFloat("discretization", "dx" , 1.0);
  dVx = configMap.getFloat("discretization", "dvx", 1.0);

  minRealx = configMap.getFloat("discretization", "minRealx", 0);
  maxRealx = configMap.getFloat("discretization", "maxRealx", nx*dx + minRealx);
  minRealVx= configMap.getFloat("discretization", "minRealVx", 0);
  
  realWidthx = maxRealx - minRealx;
  inv_dx     = 1/dx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParams::print()
{
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  std::cout << "kernelImpl : " << strKernelImpl << std::endl;
  printf( "gpu        : %d\n", gpu);
  printf( "maxIter    : %zu\n", maxIter);
  printf( "nx         : %zu\n", nx);
  printf( "nVx        : %zu\n", nVx);
  printf( "dt         : %f\n", dt);
  printf( "dx         : %f\n", dx);
  printf( "dVx        : %f\n", dVx);
  printf( "minRealx   : %f\n", minRealx );
  printf( "maxRealx   : %f\n", maxRealx);
  printf( "minRealVx  : %f\n", minRealVx);
  printf( "\n");

} // ADVParams::print
