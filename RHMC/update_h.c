// -----------------------------------------------------------------
// Update the momentum matrices
#include "so4_includes.h"
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Update the momenta with the (fairly trivial) scalar force
double scalar_force(Real eps) {
  register int i;
  register site *s;
  msg_tag *tag[2*NDIMS];
  antisym tempas,tempas_dir,tempas_opp;
  double returnit = 0.0;

  // Just subtract sigma from the momenta and compute average force
  /*FORALLSITES(i, s) {
    scalar_mult_dif_as(&(s->sigma), eps, &(mom[i]));
    returnit += magsq_as(&(s->sigma));
  }*/
  
  //Clear the force accumulator
    FORALLSITES(i, s)
       clear_as(&(force[i]));
  // The force from the kinetic term is d'Alembertian acting on sigma field 
  // To compute the force contribution from kinetic term we require two gather calls
  for(dir = XUP; dir <= TUP; dir++){
   tag[dir] = start_gather_site(F_OFFSET(sigma), sizeof(antisym), dir,
                                    EVENANDODD, gen_pt[dir]);
   tag[OPP_DIR(dir)] = start_gather_site(F_OFFSET(sigma), sizeof(antisym),OPP_DIR(dir),
                                    EVENANDODD, gen_pt[OPP_DIR(dir)]);
  }
  for(dir = XUP; dir <=TUP ; dir++){
   wait_gather(tag[dir]);
   wait_gather(tag[OPP_DIR(dir)]);

   FORALLSITES(i,s){
   as_copy((antisym *) gen_pt[dir][i],&tempas_dir);
   as_copy((antisym *) gen_pt[OPP_DIR(dir)][i],&tempas_opp);
   add_as(&tempas_dir,&tempas_opp,&tempas);
   scalar_mult_add_as(&force[i],&tempas,-1.0*kappa,&force[i]);

   }

  }
  // Add the contribution from on-site term 
  // Update the momenta from the scalar force
  FORALLSITES(i,s){
    scalar_mult_add_as(&force[i],&(s->sigma),(1.0+2*kappa*NDIMS),&force[i]);
    scalar_mult_add_as(&(mom[i]),&(force[i]),tr,&(mom[i]));
    returnit += magsq_as(&force[i]);
  }

  g_doublesum(&returnit);
  return (eps * sqrt(returnit) / volume);
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Update the momenta with the fermion force
//   -F_{ab} = Ms^a s^b + 0.5 eps_{abcd} Ms^c s^d
// "s" is sol while "Ms" is tempvec
// Assume that the multiCG has been run, with the solution vectors in sol[j]
// Accumulate force for each pole, using tempvec for temporary storage
double fermion_force(Real eps, vector *src, vector **sol) {
#if (DIMF != 4)
  #error "Assuming DIMF=4!"
#endif
  register int i;
  register site *s;
  int n, a, b, c, d, p;
  Real tr;
  double returnit = 0.0;
  antisym tempas;

  // Clear the force accumulators
  FORALLSITES(i, s)
    clear_as(&(force[i]));

  // Accumulate vector product in force
  for (n = 0; n < Norder; n++) {
    fermion_op(sol[n], tempvec, PLUS);
    FORALLSITES(i, s) {
      // Makes sense to multiply here by amp2[n]...
      scalar_mult_vec(&(tempvec[i]), amp2[n], &(tempvec[i]));

      clear_as(&tempas);
      for (a = 0; a < DIMF; a++) {
        for (b = a + 1; b < DIMF; b++) {
          p = as_index[a][b];
          tempas.e[p] += tempvec[i].c[a] * sol[n][i].c[b];
          tempas.e[p] -= tempvec[i].c[b] * sol[n][i].c[a];
          for (c = 0; c < DIMF; c++) {
            for (d = c + 1; d < DIMF; d++) {
              tr = perm[a][b][c][d];    // No half since not double-counting
              tempas.e[p] += tr * tempvec[i].c[c] * sol[n][i].c[d];
              tempas.e[p] -= tr * tempvec[i].c[d] * sol[n][i].c[c];
            }
          }
        }
      }
      scalar_mult_sum_as(&tempas, G, &(force[i]));
    }
  }

  // Update the momenta from the fermion force
  // Note opposite sign compared to gauge force...
  FORALLSITES(i, s) {
    scalar_mult_sum_as(&(force[i]), eps, &(mom[i]));
    returnit += magsq_as(&(force[i]));
  }
  g_doublesum(&returnit);
  return (eps * sqrt(returnit) / volume);
}
// -----------------------------------------------------------------
