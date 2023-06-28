#include "InitParams.h"

// ======================================================
// ======================================================
void InitParams::setup(const ConfigMap& configMap)
{
  // run parameters
  gpu = configMap.getBool("run", "gpu", false);
  outputSolution = configMap.getBool("run", "outputSolution", false);

  kernelImpl = configMap.getString("run", "kernelImpl", "BasicRange3D");
} // InitParams::setup

// ======================================================
// ======================================================
void InitParams::print()
{
  print_kokkos();
  printf( "gpu        : %d\n", gpu);
  printf( "\n");

} // InitParams::print

// ======================================================
// ======================================================
void InitParams::print_kokkos()
{
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  std::cout << "kernelImpl : " << kernelImpl << std::endl;
} // InitParams::print_kokkos