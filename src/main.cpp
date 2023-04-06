#include <iostream>
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include "utils.h"
#include <advectors.h>

bool static constexpr _DEBUG = false;

// ==========================================
// ==========================================
sycl::event advection(
   sycl::queue &Q,
   sycl::buffer<double, 2> &buff_fdistrib,
   const ADVParams &params)
{
   // AdvX::BasicRange advector;
   // AdvX::Hierarchical advector;
   // AdvX::NDRange advector;
   AdvX::Scoped advector;

   int static const maxIter = params.maxIter;
 
   // Time loop, cannot parallelize this
   for(int t=0; t < maxIter; ++t){

      if(t == maxIter-1) //If it's last iteration, we return an event to wait
         return advector(Q, buff_fdistrib, params);

      advector(Q, buff_fdistrib, params);
      
      if(_DEBUG){
         std::cout << "\nFdist_p" << t << " :" << std::endl;
         print_buffer(buff_fdistrib, params);
      }
   } // end for t < T

   //unused code, here to remove warning about non-void function not returning
   return advection(Q, buff_fdistrib, params);
} // end advection

// ==========================================
// ==========================================
int main(int argc, char** argv) {
    /* Read input parameters */
    std::string input_file = argc>1 ? std::string(argv[1]) : "advection.ini";
    ConfigMap configMap(input_file);
    ADVParams params = ADVParams();
    params.setup(configMap);
    params.print();
    const auto nx = params.nx;
    const auto nVx = params.nVx;
    const auto maxIter = params.maxIter;
    
    const auto run_on_gpu = params.gpu;

    /* Use different queues depending on SYCL implem */
#ifdef __INTEL_LLVM_COMPILER
   std::cout << "Running with DPCPP" << std::endl;
   /* Double not supported on IntelGraphics so we choose the CPU
   if not with OpenSYCL */
   sycl::queue Q{sycl::cpu_selector_v};
#else //__HIPSYCL__
   std::cout << "Running with OpenSYCL (hipSYCL)" << std::endl;

   sycl::device d;
   if(run_on_gpu)
      d = sycl::device{sycl::gpu_selector_v};
   else
      d = sycl::device{sycl::cpu_selector_v};
      
      sycl::queue Q{d};
#endif

    /* Display infos on current device */
   std::cout << "Using device: "
                << Q.get_device().get_info<sycl::info::device::name>()
                << "\n";

   /* Buffer for the distribution function containing the probabilities of 
   having a particle at a particular speed and position */
   sycl::buffer<double, 2> buff_fdistrib(sycl::range<2>(params.nVx, params.nx));
   fill_buffer(Q, buff_fdistrib, params);
   
   if(_DEBUG){
      std::cout << "Fdist:" << std::endl;
      print_buffer(buff_fdistrib, params);
   }

   auto start = std::chrono::high_resolution_clock::now();
   advection(Q, buff_fdistrib, params).wait_and_throw();
   auto end = std::chrono::high_resolution_clock::now();

   auto res = check_result(Q, buff_fdistrib, params, _DEBUG);
   std::cout << "\nSqrt_sum: " << res << std::endl;

   std::chrono::duration<double> elapsed_seconds = end-start;
   std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";
   std::cout << "upd_cells_per_sec: "
        << ((nVx*nx*maxIter)/elapsed_seconds.count())/1e6 << " Mcells/sec\n";
   std::cout << "parsing;" << nVx*nx << ";" << nx << ";" << nVx << std::endl;
   return 0;
}
