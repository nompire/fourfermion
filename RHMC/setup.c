// -----------------------------------------------------------------
// Four-fermion SO(4) system setup
#include "so4_includes.h"

#define IF_OK if(status==0)

// Each node has a params structure for passing simulation parameters
#include "params.h"
params par_buf;
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// On node zero, read lattice size and seed, and send to others
int initial_set() {
  int prompt = 0, status = 0;
  if (mynode() == 0) {
    // Print banner
    // stringification kludge from GNU preprocessor manual
    // http://gcc.gnu.org/onlinedocs/cpp/Stringification.html
#define XSTR(s) STR(s)
#define STR(s) #s
    // end kludge
    printf("Four-fermion SO(%d) system\n", DIMF);
    printf("Microcanonical simulation with refreshing\n");
    printf("Machine = %s, with %d nodes\n", machine_type(), numnodes());
#ifdef HMC_ALGORITHM
    printf("Hybrid Monte Carlo algorithm\n");
#endif
#ifdef PHI_ALGORITHM
    printf("Phi algorithm\n");
#else   // Quit!
    printf("Only works for phi algorithm\n");
    exit(1);
#endif
    time_stamp("start");
    status = get_prompt(stdin,  &prompt);

    IF_OK status += get_i(stdin, prompt, "nx", &par_buf.nx);
    IF_OK status += get_i(stdin, prompt, "ny", &par_buf.ny);
    IF_OK status += get_i(stdin, prompt, "nz", &par_buf.nz);
    IF_OK status += get_i(stdin, prompt, "nt", &par_buf.nt);
    IF_OK status += get_i(stdin, prompt, "PBC", &par_buf.PBC);
    IF_OK status += get_i(stdin, prompt, "iseed", &par_buf.iseed);

    // Number of Nth roots to take (in addition to 1/4 power)
    IF_OK status += get_i(stdin, prompt, "Nroot", &par_buf.Nroot);

    // RHMC degree
    IF_OK status += get_i(stdin, prompt, "Norder", &par_buf.Norder);

    if (status > 0)
      par_buf.stopflag = 1;
    else
      par_buf.stopflag = 0;

    if (par_buf.PBC >= 0)
      printf("Periodic temporal boundary conditions\n");
    if (par_buf.PBC < 0)
      printf("Antiperiodic temporal boundary conditions\n");
  }

  // Broadcast parameter buffer from node 0 to all other nodes
  broadcast_bytes((char *)&par_buf, sizeof(par_buf));
  if (par_buf.stopflag != 0)
    normal_exit(0);

  nx = par_buf.nx;
  ny = par_buf.ny;
  nz = par_buf.nz;
  nt = par_buf.nt;
  PBC = par_buf.PBC;
  iseed = par_buf.iseed;

  // Set up stuff for RHMC and multi-mass CG
  Nroot = par_buf.Nroot;
  fnorm = malloc(Nroot * sizeof(fnorm));
  max_ff = malloc(Nroot * sizeof(max_ff));

  Norder = par_buf.Norder;
  amp = malloc(Norder * sizeof(amp));
  amp2 = malloc(Norder * sizeof(amp2));
  amp4 = malloc(Norder * sizeof(amp4));
  shift = malloc(Norder * sizeof(shift));
  shift2 = malloc(Norder * sizeof(shift2));
  shift4 = malloc(Norder * sizeof(shift4));

  this_node = mynode();
  number_of_nodes = numnodes();
  volume = nx * ny * nz * nt;
  total_iters = 0;
  return prompt;
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Set up Kogut--Susskind phase factors, sum_{nu < mu} (-1)^{i[nu]}
void setup_phases() {
  register int i;
  register site *s;

  FORALLSITES(i, s) {
    s->phase[TUP] = 1.0;
    if ((s->t) % 2 == 1)
      s->phase[XUP] = -1.0;
    else
      s->phase[XUP] = 1.0;

    if ((s->x) % 2 == 1)
      s->phase[YUP] = -s->phase[XUP];
    else
      s->phase[YUP] = s->phase[XUP];

    if ((s->y) % 2 == 1)
      s->phase[ZUP] = -s->phase[YUP];
    else
      s->phase[ZUP] = s->phase[YUP];
  }
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Allocate space for fields
void make_fields() {
  double size = (double)(2.0 * sizeof(vector));
  FIELD_ALLOC(src, vector);
  FIELD_ALLOC(dest, vector);

  // Momenta and forces for the scalars
  size += (double)(2.0 * sizeof(antisym));
  FIELD_ALLOC(mom, antisym);
  FIELD_ALLOC(force, antisym);

  // Temporary vector and matrix
  size += (double)(sizeof(vector) + sizeof(antisym));
  FIELD_ALLOC(tempvec, vector);
  FIELD_ALLOC(tempas, antisym);

#ifdef CORR
  // Propagator for point-source condensates
  size += (double)(2.0 * sizeof(matrix));
  FIELD_ALLOC(prop, matrix);
  FIELD_ALLOC(prop2, matrix);
#endif

  size *= sites_on_node;
  node0_printf("Mallocing %.1f MBytes per core for fields\n", size / 1e6);
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
int setup() {
  int prompt;

  // Print banner, get volume and seed
  prompt = initial_set();
  // Initialize the node random number generator
  initialize_prn(&node_prn, iseed, volume + mynode());
  // Initialize the layout functions, which decide where sites live
  setup_layout();
  // Allocate space for lattice, set up coordinate fields
  make_lattice();
  // Set up neighbor pointers and comlink structures
  make_nn_gathers();
  // Set up phases
  setup_phases();
  // Allocate space for fields
  make_fields();

  return prompt;
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Read in parameters for Monte Carlo
int readin(int prompt) {
  // prompt=1 indicates prompts are to be given for input
  int status;
  Real x;
#ifdef CORR
  int j, k;
#endif

  // On node zero, read parameters and send to all other nodes
  if (this_node == 0) {
    printf("\n\n");
    status = 0;

    // Warms, trajecs
    IF_OK status += get_i(stdin, prompt, "warms", &par_buf.warms);
    IF_OK status += get_i(stdin, prompt, "trajecs", &par_buf.trajecs);
    IF_OK status += get_f(stdin, prompt, "traj_length", &par_buf.traj_length);

    // Number of fermion and scalar steps
    IF_OK status += get_i(stdin, prompt, "nstep", &par_buf.nsteps[0]);
    IF_OK status += get_i(stdin, prompt, "nstep_scalar", &par_buf.nsteps[1]);

    // Trajectories between propagator measurements
    IF_OK status += get_i(stdin, prompt, "traj_between_meas",
                          &par_buf.propinterval);

    // Four-fermion coupling
    IF_OK status += get_f(stdin, prompt, "G", &par_buf.G);

    // Scalar field kinetic term
    IF_OK status += get_f(stdin, prompt, "kappa", &par_buf.kappa);
 
    // On-site SO(4)-breaking mass term
    IF_OK status += get_f(stdin, prompt, "site_mass", &par_buf.site_mass);

    // SO(4)-symmetric shift-symmetry-breaking mass term
    IF_OK status += get_f(stdin, prompt, "link_mass", &par_buf.link_mass);

    // Maximum conjugate gradient iterations
    IF_OK status += get_i(stdin, prompt, "max_cg_iterations", &par_buf.niter);

    // Error per site for conjugate gradient
    IF_OK {
      status += get_f(stdin, prompt, "error_per_site", &x);
      par_buf.rsqmin = x;
    }

#ifdef CORR
    // Number of stochastic sources
    IF_OK status += get_i(stdin, prompt, "Nstoch", &par_buf.Nstoch);

    // Number of point sources
    IF_OK status += get_i(stdin, prompt, "Nsrc", &par_buf.Nsrc);
    if (par_buf.Nsrc > MAX_SRC) {
      node0_printf("Error: Can only handle up to %d sources\n", MAX_SRC);
      node0_printf("       Recompile with different MAX_SRC for more\n");
      status++;
    }
    for (j = 0; j < par_buf.Nsrc; j++) {
      IF_OK status += get_vi(stdin, prompt, "pnt", par_buf.pnts[j], NDIMS);
    }
#endif

#ifdef EIG
    // Number of eigenvalues to calculate
    IF_OK status += get_i(stdin, prompt, "Nvec", &par_buf.Nvec);
    IF_OK status += get_f(stdin, prompt, "eig_tol", &par_buf.eig_tol);
    IF_OK status += get_i(stdin, prompt, "maxIter", &par_buf.maxIter);
#endif

    // Find out what kind of starting lattice to use
    IF_OK status += ask_starting_lattice(stdin,  prompt, &par_buf.startflag,
                                         par_buf.startfile);

    // Find out what to do with lattice at end
    IF_OK status += ask_ending_lattice(stdin,  prompt, &(par_buf.saveflag),
                                       par_buf.savefile);

    if (status > 0)
      par_buf.stopflag = 1;
    else
      par_buf.stopflag = 0;
  }

  // Broadcast parameter buffer from node0 to all other nodes
  broadcast_bytes((char *)&par_buf, sizeof(par_buf));
  if (par_buf.stopflag != 0)
    normal_exit(0);

  warms = par_buf.warms;
  trajecs = par_buf.trajecs;
  traj_length = par_buf.traj_length;
  nsteps[0] = par_buf.nsteps[0];
  nsteps[1] = par_buf.nsteps[1];

  propinterval = par_buf.propinterval;
  niter = par_buf.niter;
  rsqmin = par_buf.rsqmin;

  G = par_buf.G;
  kappa = par_buf.kappa;
  site_mass = par_buf.site_mass;
  link_mass = par_buf.link_mass;
  if (G == 0.0 && site_mass != 0.0) {
    node0_printf("Warning: Setting site_mass = 0 since G = 0\n");
    site_mass = 0.0;
  }
  // Interpret negative site_mass as coefficient of staggered term
  if (site_mass < 0) {
    stagger = 1;
    site_mass *= -1;    // Need it positive, after all!
  }
  else
    stagger = -1;

#ifdef CORR
  Nstoch = par_buf.Nstoch;
  Nsrc = par_buf.Nsrc;
  for (j = 0; j < Nsrc; j++) {
    for (k = 0; k < NDIMS; k++)
      pnts[j][k] = par_buf.pnts[j][k];
  }
#endif
#ifdef EIG
  Nvec = par_buf.Nvec;
  eig_tol = par_buf.eig_tol;
  maxIter = par_buf.maxIter;
#endif

  startflag = par_buf.startflag;
  saveflag = par_buf.saveflag;
  strcpy(startfile, par_buf.startfile);
  strcpy(savefile, par_buf.savefile);

  // Do whatever is needed to get lattice
  startlat_p = reload_lattice(startflag, startfile);
  return 0;
}
// -----------------------------------------------------------------
