#include <iostream>
#include <sycl/sycl.hpp>

int    static constexpr Nx       = 4;
int    static constexpr NVx      = 1;

int    static constexpr T        = 1;     //nombre total d'iterations
double static constexpr dt       = 0.125; //durée réelle d'une iteration
double static constexpr dx       = 1;     //espacement physique entre 2 cellules du maillage
double static constexpr dvx      = 1;
double static constexpr minRealx  = 0;
double static constexpr maxRealx  = Nx*dx + minRealx;
double static constexpr minRealvx = -10;

double static constexpr inv_dx      = 1/dx;  // inverse de dx
double static constexpr realWidthx  = maxRealx - minRealx;

/* Lagrange variables, order, number of points, offset from the current point */
int    static constexpr LAG_ORDER  = 5;
int    static constexpr LAG_PTS    = 6;
int    static constexpr LAG_OFFSET = 2;

/* Computes the coefficient for semi lagrangian interp of order 5 */
inline void lag_basis(double px, double *coef) {
  constexpr double loc[] = { -1. / 24, 1. / 24., -1. / 12., 1. / 12., -1. / 24., 1. / 24. };
  const double pxm2 = px - 2.;
  const double sqrpxm2 = pxm2 * pxm2;
  const double pxm2_01 = pxm2 * (pxm2 - 1.);

  coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
  coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
  coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) * (5 * sqrpxm2 - 3 * pxm2 - 6.);
  coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) * (5 * sqrpxm2 - 7 * pxm2 - 4.);
  coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
  coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
}

/* Computes the covered distance by x during dt and returns the feet coord */
inline int displ(const int &ix, const int &ivx ){
   const double x    = minRealx  + ix * dx; //real coordinate of particles at ix
   const double vx   = minRealvx + ivx * dvx; //real speed of particles at ivx
   const double displx = dt * vx;

   const double xstar =
      minRealx + sycl::fmod(realWidthx + x - displx - minRealx, realWidthx);

   return xstar;
}

/* Fills a f distrib sycl buffer */
void fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &fdist){
  q.submit([&](sycl::handler &cgh){
    sycl::accessor FDIST(fdist, cgh, sycl::write_only, sycl::no_init);

    cgh.parallel_for(fdist.get_range(), [=](sycl::id<2> itm)
    {
      double x = itm[1];
      //Init les données avec sinus
      FDIST[0][itm[1]] = sycl::sin(x);
      // FDIST[0][itm[1]] = itm[1];
      // FDIST[1][itm[1]] = 4.5;
      // FDIST[2][itm[1]] = 5.5;
      // FDIST[3][itm[1]] = 6.5;
    });
  }).wait(); // end q.submit
}

void print_buffer(sycl::buffer<double, 2> &fdist){
  sycl::host_accessor tab(fdist, sycl::read_only);

   for(int iv = 0; iv < NVx; ++iv){
      for(int ix = 0; ix < Nx; ++ix){
         std::cout << tab[iv][ix] << " ";
      }
      std::cout << std::endl;
   }
}

int main(int, char**) {

   /* Buffer for the distribution function containing the probabilities of 
   having a particle for a particular speed and position */
   sycl::buffer<double, 2> buff_fdistrib(sycl::range<2>(NVx, Nx));
   sycl::buffer<double, 2> buff_fdistrib_p1(sycl::range<2>(NVx, Nx));

   sycl::queue Q;

   fill_buffer(Q, buff_fdistrib);
   fill_buffer(Q, buff_fdistrib_p1);
   std::cout << "Fdist :" << std::endl;
   print_buffer(buff_fdistrib);


   // Time loop, cannot parallelize this
   for(int t=0; t<T; ++t){

    Q.submit([&](sycl::handler& cgh){

      auto fdist    = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
      auto fdist_p1    = buff_fdistrib_p1.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=](){

         //For each Vx
         for(int ivx = 0; ivx < NVx; ++ivx){

            // std::array<double, Nx> ftmp{};

            //For each x with regards to current Vx
            for(int ix = 0; ix < Nx; ++ix){
               double const xFootCoord = displ(ix, ivx);

               // Corresponds to the index of the cell to the left of footCoord
               const int leftDiscreteCell = sycl::floor((xFootCoord-minRealx) * inv_dx);
         
               //d_prev1 : dist entre premier point utilisé pour l'interpolation et xFootCoord (dans l'espace de coord discret, même si double)
               const double d_prev1 = LAG_OFFSET + inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));
               

               double coef[LAG_PTS];
               lag_basis(d_prev1, coef);

               const int ipos1 = leftDiscreteCell - LAG_OFFSET;
               double ftmp = 0.;
               for(int k=0; k<=LAG_ORDER; k++) {
                  int idx_ipos1 = (Nx + ipos1 + k) % Nx; //penser à essayer de retirer ce modulo
                  // ftmp += coef[k] * fdist[ivx][idx_ipos1];
                  // ftmp += coef[k] * ftmp_input[idx_ipos1];
                  ftmp += coef[k] * fdist[ivx][idx_ipos1];
               }

               // memcpy();
               //avec ftmp un array de la taille de x,
               fdist_p1[ivx][ix] = ftmp; 
            } // end for X

         } // end for Vx
      }); // end cgh.single_task()
    }).wait_and_throw(); // Q.submit


      Q.submit([&](sycl::handler& cgh){
         auto FDIST    = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
         auto FDIST_p1 = buff_fdistrib_p1.get_access<sycl::access::mode::read>(cgh);

         cgh.copy(FDIST_p1, FDIST);
         // cgh.copy(FDIST_p1, FDIST);
      }).wait_and_throw();
   } // end for t < T

   // std::cout << "Fdist :" << std::endl;
   // print_buffer(buff_fdistrib);

   std::cout << "\nFdist_p1 :" << std::endl;
   print_buffer(buff_fdistrib_p1);

   return 0;
}