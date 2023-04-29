#include "AdvectionParams.h"

// ======================================================
// ======================================================
void ADVParams::setup(const ConfigMap& configMap)
{
  // geometry
  nx  = configMap.getInteger("geometry", "nx",  512);
  nVx = configMap.getInteger("geometry", "nVx", 1024);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange");

  // discretization parameters
  dt  = configMap.getFloat("discretization", "dt" , 0.0001);

  minRealx = configMap.getFloat("discretization", "minRealx", 0.0);
  maxRealx = configMap.getFloat("discretization", "maxRealx", 1.0);
  minRealVx= configMap.getFloat("discretization", "minRealVx", 0.0);
  maxRealVx= configMap.getFloat("discretization", "maxRealVx", 1.0);

  realWidthx = maxRealx - minRealx;
  dx = realWidthx / nx;
  dVx = (maxRealVx - minRealVx) / nVx;

  inv_dx     = 1/dx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParams::print()
{
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  std::cout << "kernelImpl : " << kernelImpl << std::endl;
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
  printf( "maxRealVx  : %f\n", maxRealVx);
  printf( "\n");

} // ADVParams::print
