#include "AdvectionParams.h"
#include <iostream>

ADVParams::ADVParams(ADVParamsNonCopyable &other){
  nx = other.nx;
  nvx = other.nvx;
  nz = other.nz;

  maxIter = other.maxIter;
  gpu = other.gpu;
  outputSolution = other.outputSolution;

  wg_size_x = other.wg_size_x;
  wg_size_y = other.wg_size_y;

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
  nvx = configMap.getInteger("geometry", "nvx", 64);
  nz = configMap.getInteger("geometry", "nz", 32);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange2D");
  wg_size_x = configMap.getInteger("run", "workGroupSizeX", 128);
  wg_size_y = configMap.getInteger("run", "workGroupSizeY", 1);

  // discretization parameters
  dt  = configMap.getFloat("discretization", "dt" , 0.0001);

  minRealX = configMap.getFloat("discretization", "minRealX", 0.0);
  maxRealX = configMap.getFloat("discretization", "maxRealX", 1.0);
  minRealVx= configMap.getFloat("discretization", "minRealVx", 0.0);
  maxRealVx= configMap.getFloat("discretization", "maxRealVx", 1.0);

  update_deltas();
  // realWidthX = maxRealX - minRealX;
  // dx = realWidthX / nx;
  // dvx = (maxRealVx - minRealVx) / nvx;

  // inv_dx     = 1/dx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParams::update_deltas()
{
  realWidthX = maxRealX - minRealX;
  dx = realWidthX / nx;
  dvx = (maxRealVx - minRealVx) / nvx;

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
  std::cout << "wgSizeX     : " << wg_size_x    << std::endl;
  std::cout << "wgSizeY     : " << wg_size_y    << std::endl;
  printf( "gpu        : %d\n", gpu);
  printf( "maxIter    : %zu\n", maxIter);
  printf( "nvx        : %zu\n", nvx);
  printf( "nx         : %zu\n", nx);
  printf( "nz         : %zu\n", nz);
  printf( "dt         : %f\n", dt);
  printf( "dx         : %f\n", dx);
  printf( "dvx        : %f\n", dvx);
  printf( "minRealX   : %f\n", minRealX );
  printf( "maxRealX   : %f\n", maxRealX);
  printf( "minRealVx  : %f\n", minRealVx);
  printf( "maxRealVx  : %f\n", maxRealVx);
  printf( "\n");

} // ADVParams::print
