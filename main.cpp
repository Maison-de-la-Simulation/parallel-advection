#include <iostream>
#include <sycl/sycl.hpp>
bool static constexpr _DEBUG = false;

size_t    static constexpr Nx       = 512;
size_t    static constexpr NVx      = 4;

size_t    static constexpr T        = Nx/2;     //nombre total d'iterations
double static constexpr dt       = 1; //durée réelle d'une iteration
double static constexpr dx       = 1;     //espacement physique entre 2 cellules du maillage
double static constexpr dvx      = 1;
double static constexpr minRealx  = 0;
double static constexpr maxRealx  = Nx*dx + minRealx;
double static constexpr minRealvx = 0;

double static constexpr inv_dx      = 1/dx;  // inverse de dx
double static constexpr realWidthx  = maxRealx - minRealx;

/* Lagrange variables, order, number of points, offset from the current point */
int    static constexpr LAG_ORDER  = 5;
int    static constexpr LAG_PTS    = 6;
int    static constexpr LAG_OFFSET = 2;

// ==========================================
// ==========================================
/* Computes the coefficient for semi lagrangian interp of order 5 */
void
lag_basis(const double &px, double* coef){
    constexpr double loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                              1. / 12., -1. / 24., 1. / 24.};
    const double pxm2 = px - 2.;
    const double sqrpxm2 = pxm2 * pxm2;
    const double pxm2_01 = pxm2 * (pxm2 - 1.);

    coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
    coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
    coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) *
              (5 * sqrpxm2 - 3 * pxm2 - 6.);
    coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) *
              (5 * sqrpxm2 - 7 * pxm2 - 4.);
    coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
    coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
} // end lag_basis

// ==========================================
// ==========================================
/* Computes the covered distance by x during dt and returns the feet coord */
int
displ(const int &ix, const int &ivx ){
   const double x    = minRealx  + ix * dx; //real coordinate of particles at ix
   const double vx   = minRealvx + ivx * dvx; //real speed of particles at ivx
   const double displx = dt * vx;

   const double xstar =
      minRealx + sycl::fmod(realWidthx + x - displx - minRealx, realWidthx);

   return xstar;
} // end displ

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &fdist){
  q.submit([&](sycl::handler &cgh){
    sycl::accessor FDIST(fdist, cgh, sycl::write_only, sycl::no_init);

    cgh.parallel_for(fdist.get_range(), [=](sycl::id<2> itm)
    {
      double x = itm[1];
      for(int ivx = 0; ivx < NVx; ++ivx){
         FDIST[ivx][itm[1]] = ivx % 2 == 0 ? sycl::sin(x) : sycl::cos(x);
      }
    });
  }).wait(); // end q.submit
}

// ==========================================
// ==========================================
void
print_buffer(sycl::buffer<double, 2> &fdist){
  sycl::host_accessor tab(fdist, sycl::read_only);

   for(int iv = 0; iv < NVx; ++iv){
      for(int ix = 0; ix < Nx; ++ix){
         std::cout << tab[iv][ix] << " ";
      }
      std::cout << std::endl;
   }
} // end print_buffer

// ==========================================
// ==========================================
double
check_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib){
   /* Fill a buffer the same way we filled fdist at init */
   sycl::buffer<double, 2> buff_init(sycl::range<2>(NVx, Nx));
   fill_buffer(Q, buff_init);

   if(_DEBUG){
      std::cout << "\nFdist_init :" << std::endl;
      print_buffer(buff_init);
   }

   /* Check norm of difference, should be 0 */
   sycl::buffer<double, 2> buff_res(buff_init.get_range());
   Q.submit([&](sycl::handler& cgh){
      auto A = buff_init.get_access<sycl::access::mode::read>(cgh);
      auto B = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
      sycl::accessor C(buff_res, cgh, sycl::write_only, sycl::no_init);

      cgh.parallel_for(buff_init.get_range(), [=](auto itm){
         C[itm] = A[itm] - B[itm];
         C[itm] *= C[itm]; // We square each elements
      });
   }).wait_and_throw();

   if(_DEBUG){
      std::cout << "\nDifference Buffer :" << std::endl;
      print_buffer(buff_res);
   }

   double sumResult = 0;
   {
      sycl::buffer<double> buff_sum { &sumResult, 1 };

      Q.submit([&](sycl::handler& cgh) {
      // Input values to reductions are standard accessors
      auto inputValues = buff_res.get_access<sycl::access_mode::read>(cgh);

#ifdef __INTEL_LLVM_COMPILER //for DPCPP
      auto sumReduction = sycl::reduction(buff_sum, cgh, sycl::plus<>());
#else //for openSYCL
      auto sumAcc = buff_sum.get_access<sycl::access_mode::write>(cgh);
      auto sumReduction = sycl::reduction(sumAcc, sycl::plus<double>());
#endif
      cgh.parallel_for(buff_res.get_range(), sumReduction,
         [=](auto idx, auto& sum) {
            // plus<>() corresponds to += operator, so sum can be
            // updated via += or combine()
            sum += inputValues[idx];
         });
      }).wait_and_throw();
   }

   return std::sqrt(sumResult);
} // end check_result

// ==========================================
// ==========================================
void advection(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib){

   const sycl::range<1> nb_wg{NVx};
   const sycl::range<1> wg_size{Nx};

   /* Cannot use local memory with basic range parallel_for so I use a global
   buffer of size NVx * Nx*/
   sycl::buffer<double, 2> global_buff_ftmp(sycl::range<2>(NVx, Nx));

   // Time loop, cannot parallelize this
   for(int t=0; t<T; ++t){

    Q.submit([&](sycl::handler& cgh){

      auto fdist_write    = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
      auto fdist_read     = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
      // auto fdist_p1    = buff_fdistrib_p1.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=](){

         //For each Vx
         for(int ivx = 0; ivx < NVx; ++ivx){

            // std::array<double, Nx> ftmp{};
            // double* x_slice = sycl::malloc_device(sizeof(double)*Nx,);

            // Problem here is that with thie method 
            // we have to set the accessor in read_write mode instead of 2 accessors, one in read, one in write
            double slice_x[Nx];
            memcpy(slice_x, fdist_read.get_pointer()+ivx*Nx, Nx*sizeof(double));

            //For each x with regards to current Vx
            for(int ix = 0; ix < Nx; ++ix){
               double const xFootCoord = displ(ix, ivx);

               // Corresponds to the index of the cell to the left of footCoord
               const int leftDiscreteCell = sycl::floor((xFootCoord-minRealx) * inv_dx);
         
               //d_prev1 : dist entre premier point utilisé pour l'interpolation et xFootCoord (dans l'espace de coord discret, même si double)

               /* Percentage of the distance inside the cell ???? TODO : Find better var name */
               const double d_prev1 = LAG_OFFSET + inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

               double coef[LAG_PTS];
               lag_basis(d_prev1, coef);

               const int ipos1 = leftDiscreteCell - LAG_OFFSET;
               double ftmp = 0.;
               for(int k=0; k<=LAG_ORDER; k++) {
                  int idx_ipos1 = (Nx + ipos1 + k) % Nx; //penser à essayer de retirer ce modulo. Possible si on a une distance max on alloue un tableau avec cette distance max en plus des deux côtés

                  //Pour faire in place il faut utiliser un buffer de taille Nx. Soit on s'en sert en buffer d'input pour la lecture et on met la valeur directement dans fdist
                  // soit on s'en sert en buffer output : on stocke le résultat dedans puis on copie tout ce buffer dans la ligne correspondante dans fdist

                  /* Ici en utilisant fdist en lecture */
                  // ftmp += coef[k] * fdist[ivx][idx_ipos1];

                  /* Ici en utilisant slice_x as an input */
                  ftmp += coef[k] * slice_x[idx_ipos1];

                  /*  */
                  // ftmp[idx_pos1] += coef[k] * fdist[ivx][idx_ipos1];
               }

               // fdist_write[ivx][ix] = ftmp; 
               fdist_write[ivx][ix] = ftmp; 
            } // end for X

         } // end for Vx
      }); // end cgh.single_task()
    }).wait_and_throw(); // end Q.submit

    if(_DEBUG){
        std::cout << "\nFdist_p" << t << " :" << std::endl;
        print_buffer(buff_fdistrib);
    }
   } // end for t < T
} // end advection

// ==========================================
// ==========================================
int main(int, char**) {
#ifdef __INTEL_LLVM_COMPILER
   std::cout << "Running with DPCPP" << std::endl;
   /* Double not supported on IntelGraphics so we choose the CPU
   if not with OpenSYCL */
   sycl::queue Q{sycl::cpu_selector_v};
#else //__HIPSYCL__
   // std::cout << "Running with OpenSYCL" << std::endl;
   sycl::queue Q;
#endif

   std::cout << "Running on "
                << Q.get_device().get_info<sycl::info::device::name>()
                << "\n";

   /* Buffer for the distribution function containing the probabilities of 
   having a particle for a particular speed and position */
   sycl::buffer<double, 2> buff_fdistrib(sycl::range<2>(NVx, Nx));
   fill_buffer(Q, buff_fdistrib);
   
   if(_DEBUG){
      std::cout << "Fdist:" << std::endl;
      print_buffer(buff_fdistrib);
   }

   auto start = std::chrono::high_resolution_clock::now();
   advection(Q, buff_fdistrib);
   auto end = std::chrono::high_resolution_clock::now();

   auto res = check_result(Q, buff_fdistrib);
   std::cout << "\nSqrt sum: " << res << std::endl;

   std::chrono::duration<double> elapsed_seconds = end-start;
   std::cout << "elapsed_time: " << elapsed_seconds.count() << "s\n";
   std::cout << "upd_cells_per_sec: "
        << ((NVx*Nx*T)/elapsed_seconds.count())/1e3 << " Kcells/sec\n";

   return 0;
}