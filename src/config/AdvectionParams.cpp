#include "AdvectionParams.h"
#include <iostream>

ADVParams::ADVParams(ADVParamsNonCopyable &other){
  nx = other.nx;
  nb = other.nb;
  ns = other.ns;

  maxIter = other.maxIter;
  gpu = other.gpu;
  outputSolution = other.outputSolution;

  wg_size_x = other.wg_size_x;
  wg_size_b = other.wg_size_b;

  dt = other.dt;

  minRealX   = other.minRealX;
  maxRealX   = other.maxRealX;
  minRealVx  = other.minRealVx;
  maxRealVx  = other.maxRealVx;

  realWidthX = other.realWidthX;
  dx         = other.dx        ;
  dvx        = other.dvx       ;

  inv_dx     = other.inv_dx;
};

ADVParamsNonCopyable::ADVParamsNonCopyable(ADVParams &other){
  nx = other.nx;
  nb = other.nb;
  ns = other.ns;

  maxIter = other.maxIter;
  gpu = other.gpu;
  outputSolution = other.outputSolution;

  wg_size_x = other.wg_size_x;
  wg_size_b = other.wg_size_b;

  dt = other.dt;

  minRealX   = other.minRealX;
  maxRealX   = other.maxRealX;
  minRealVx  = other.minRealVx;
  maxRealVx  = other.maxRealVx;

  realWidthX = other.realWidthX;
  dx         = other.dx        ;
  dvx        = other.dvx       ;

  inv_dx     = other.inv_dx;
};

// ======================================================
// ======================================================
void ADVParamsNonCopyable::setup(const ConfigMap& configMap)
{
  // geometry
  nx  = configMap.getInteger("geometry", "nx",  1024);
  nb = configMap.getInteger("geometry", "nb", 64);
  ns = configMap.getInteger("geometry", "ns", 32);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange2D");
  wg_size_x = configMap.getInteger("run", "workGroupSizeX", 128);
  wg_size_b = configMap.getInteger("run", "workGroupSizeB", 1);

  // discretization parameters
  dt  = configMap.getFloat("discretization", "dt" , 0.0001);

  minRealX = configMap.getFloat("discretization", "minRealX", 0.0);
  maxRealX = configMap.getFloat("discretization", "maxRealX", 1.0);
  minRealVx= configMap.getFloat("discretization", "minRealVx", 0.0);
  maxRealVx= configMap.getFloat("discretization", "maxRealVx", 1.0);

  update_deltas();
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParams::update_deltas()
{
  realWidthX = maxRealX - minRealX;
  dx = realWidthX / nx;
  dvx = (maxRealVx - minRealVx) / nb;

  inv_dx     = 1/dx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParamsNonCopyable::print()
{
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  std::cout << "kernelImpl : " << kernelImpl << std::endl;
  std::cout << "wgSizeX    : " << wg_size_x    << std::endl;
  std::cout << "wgSizeY    : " << wg_size_b    << std::endl;
  printf( "gpu        : %d\n", gpu);
  printf( "maxIter    : %zu\n", maxIter);
  printf( "nb (nvx)   : %zu\n", nb);
  printf( "nx         : %zu\n", nx);
  printf( "ns         : %zu\n", ns);
  printf( "dt         : %f\n", dt);
  printf( "dx         : %f\n", dx);
  printf( "dvx        : %f\n", dvx);
  printf( "minRealX   : %f\n", minRealX );
  printf( "maxRealX   : %f\n", maxRealX);
  printf( "minRealVx  : %f\n", minRealVx);
  printf( "maxRealVx  : %f\n", maxRealVx);
  printf( "\n");

} // ADVParams::print
