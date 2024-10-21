#include "AdvectionParams.h"
#include <iostream>

ADVParams::ADVParams(ADVParamsNonCopyable &other){
  n1 = other.n1;
  n0 = other.n0;
  n2 = other.n2;

  maxIter = other.maxIter;
  gpu = other.gpu;
  outputSolution = other.outputSolution;

  wg_size_1 = other.wg_size_1;
  wg_size_0 = other.wg_size_0;

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
  n1 = other.n1;
  n0 = other.n0;
  n2 = other.n2;

  maxIter = other.maxIter;
  gpu = other.gpu;
  outputSolution = other.outputSolution;

  wg_size_1 = other.wg_size_1;
  wg_size_0 = other.wg_size_0;

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
  n1  = configMap.getInteger("geometry", "n1",  1024);
  n0 = configMap.getInteger("geometry", "n0", 64);
  n2 = configMap.getInteger("geometry", "n2", 32);

  // run parameters
  maxIter = configMap.getInteger("run", "maxIter", 1000);
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);
  percent_loc = configMap.getFloat("run", "percent_loc", 1.0);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange");
  wg_size_1 = configMap.getInteger("run", "workGroupSizeX", 128);
  wg_size_0 = configMap.getInteger("run", "workGroupSizeY", 1);

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
  dx = realWidthX / n1;
  dvx = (maxRealVx - minRealVx) / n0;

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
  std::cout << "wgSizeX    : " << wg_size_1    << std::endl;
  std::cout << "wgSizeY    : " << wg_size_0    << std::endl;
  std::cout << "gpu        : " << gpu    << std::endl;
  std::cout << "maxIter    : " << maxIter    << std::endl;
  std::cout << "n0 (nvx)   : " << n0    << std::endl;
  std::cout << "n1         : " << n1    << std::endl;
  std::cout << "n2        : " << n2    << std::endl;
  std::cout << "percent_loc: " << percent_loc << std::endl;
  std::cout << "dt         : " << dt    << std::endl;
  std::cout << "dx         : " << dx    << std::endl;
  std::cout << "dvx        : " << dvx    << std::endl;
  std::cout << "minRealX   : " << minRealX    << std::endl;
  std::cout << "maxRealX   : " << maxRealX    << std::endl;
  std::cout << "minRealVx  : " << minRealVx    << std::endl;
  std::cout << "maxRealVx  : " << maxRealVx    << std::endl;
  std::cout << std::endl;

} // ADVParams::print
