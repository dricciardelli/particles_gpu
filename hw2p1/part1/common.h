#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <vector>
#include <list>

inline int min( int a, int b ) { return a < b ? a : b; }
inline int max( int a, int b ) { return a > b ? a : b; }

//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

//
//  timing routines
//
double read_timer( );

//
//  simulation routines
//
void set_size( int n );
double J_get_size();
double J_get_cutoff();
void init_particles( int n, particle_t *p );
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg);
void move( particle_t &p );

void J_init_particles( int n, std::vector<std::vector<std::list<particle_t> > > &p );
void J_MPI_init_particles( int n, std::vector<std::vector<std::list<particle_t> > > &p,int num_bin_x ,int num_bin_y, int seed );

//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );
void J_save( FILE *f, int n, std::vector<std::vector<std::list<particle_t> > > p );
void J_MPI_save( FILE *f, int n, std::vector<std::vector<std::list<particle_t> > > &p, int rank, int region_num_bin_x, int region_num_bin_y );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

#endif
