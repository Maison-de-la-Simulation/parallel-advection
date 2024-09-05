#include "AdvectionParams.h"
#include <iostream>

ADVParams::ADVParams(ADVParamsNonCopyable &other){
  nx = other.nx;
  ny = other.ny;
  ny1 = other.ny1;

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

ADVParamsNonCopyable::ADVParamsNonCopyable(ADVParams &other){
  nx = other.nx;
  ny = other.ny;
  ny1 = other.ny1;

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
  ny = configMap.getInteger("geometry", "ny", 64);
  ny1 = configMap.getInteger("geometry", "ny1", 32);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange");
  wg_size_x = configMap.getInteger("run", "workGroupSizeX", 128);
  wg_size_y = configMap.getInteger("run", "workGroupSizeY", 1);

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
  dvx = (maxRealVx - minRealVx) / ny;

  inv_dx     = 1/dx;
} // ADVParams::setup

// ======================================================
// ======================================================
void ADVParamsNonCopyable::print()
{
  std::cout << "##########################" << std::endl;
  std::cout << "Runtime parameters:" << std::endl;
  std::cout << "##########################" << std::endl;
  std::cout << "kernelImpl : " << kernelImpl << std::endl;
  std::cout << "wgSizeX    : " << wg_size_x    << std::endl;
  std::cout << "wgSizeY    : " << wg_size_y    << std::endl;
  std::cout << "gpu        : " << gpu    << std::endl;
  std::cout << "maxIter    : " << maxIter    << std::endl;
  std::cout << "ny (nvx)   : " << ny    << std::endl;
  std::cout << "nx         : " << nx    << std::endl;
  std::cout << "ny1         : " << ny1    << std::endl;
  std::cout << "dt         : " << dt    << std::endl;
  std::cout << "dx         : " << dx    << std::endl;
  std::cout << "dvx        : " << dvx    << std::endl;
  std::cout << "minRealX   : " << minRealX    << std::endl;
  std::cout << "maxRealX   : " << maxRealX    << std::endl;
  std::cout << "minRealVx  : " << minRealVx    << std::endl;
  std::cout << "maxRealVx  : " << maxRealVx    << std::endl;
  std::cout << std::endl;

} // ADVParams::print
