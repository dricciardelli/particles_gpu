#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "common.h"
#include <iostream>

#include <vector>
#include <list>

double size;
int num_bin_x, num_bin_y;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  timer
//
double J_get_size()
{
  return size;
}

double J_get_cutoff()
{
  return cutoff;
}

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = drand48()*2-1;
        p[i].vy = drand48()*2-1;
    }
    free( shuffle );
}

void J_init_particles( int n, std::vector<std::vector<std::list<particle_t> > > &p )
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    num_bin_x = (int)floor(size/(cutoff*1.00000001));
    num_bin_y = (int)floor(size/(cutoff*1.00000001));
    //std::cout << "num bin_x: "<<num_bin_x << std::endl;
    //std::cout << "num bin_y: "<<num_bin_y << std::endl;
    p.resize(num_bin_x);

    for (int i = 0; i<num_bin_x;i++)
    {
      p[i].resize(num_bin_y);
      // for (int j =0; j<num_bin_y;j++)
      //     p[i][j] = std::list<particle_t>();
    }



    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    int bin_x, bin_y;
    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        //std::cout << "Here2a" << std::endl;
        bin_x = (int)(((1.+(k%sx))/(1+sx))*(num_bin_x));
        //std::cout << "bin_x: "<<bin_x << std::endl;
        //std::cout << "bin_y: "<<bin_y << std::endl;
        //std::cout << "Here2b" << std::endl;
        bin_y = (int)(((1.+(k/sx))/(1+sy))*(num_bin_y));
        particle_t insert_particle;
        insert_particle.x = size*(1.+(k%sx))/(1+sx);
        insert_particle.y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        insert_particle.vx = drand48()*2-1;
        insert_particle.vy = drand48()*2-1;
        //std::cout << "Here2c" << std::endl;
        p[bin_x][bin_y].push_back(insert_particle);
        //std::cout << "Here2d" << std::endl;
    }
    //std::cout << "Here2e" << std::endl;
    free( shuffle );
    //std::cout << "Here2f" << std::endl;
}

void J_MPI_init_particles( int n, std::vector<std::vector<std::list<particle_t> > > &p,int num_bin_x ,int num_bin_y, int seed )
{
    srand48( seed ); //use the same set of particles

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    //std::cout << "num bin_x: "<<num_bin_x << std::endl;
    //std::cout << "num bin_y: "<<num_bin_y << std::endl;
    p.resize(num_bin_x);

    for (int i = 0; i<num_bin_x;i++)
    {
      p[i].resize(num_bin_y);
      // for (int j =0; j<num_bin_y;j++)
      //     p[i][j] = std::list<particle_t>();
    }



    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    int bin_x, bin_y;
    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        //std::cout << "Here2a" << std::endl;
        bin_x = (int)(((1.+(k%sx))/(1+sx))*(num_bin_x));
        //std::cout << "bin_x: "<<bin_x << std::endl;
        //std::cout << "bin_y: "<<bin_y << std::endl;
        //std::cout << "Here2b" << std::endl;
        bin_y = (int)(((1.+(k/sx))/(1+sy))*(num_bin_y));
        particle_t insert_particle;
        insert_particle.x = size*(1.+(k%sx))/(1+sx);
        insert_particle.y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        insert_particle.vx = drand48()*2-1;
        insert_particle.vy = drand48()*2-1;
        //std::cout << "Here2c" << std::endl;
        p[bin_x][bin_y].push_back(insert_particle);
        //std::cout << "Here2d" << std::endl;
    }
    //std::cout << "Here2e" << std::endl;
    free( shuffle );
    //std::cout << "Here2f" << std::endl;
}

//
//  interact two particles
//
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg)
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
	   if (r2 != 0)
        {
    	   if (r2/(cutoff*cutoff) < *dmin * (*dmin))
    	    *dmin = sqrt(r2)/cutoff;

          (*davg) += sqrt(r2)/cutoff;
          (*navg) ++;
        }

    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );



    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
//  integrate the ODE
//
void move( particle_t &p )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  I/O routines
//
void J_save( FILE *f, int n, std::vector<std::vector<std::list<particle_t> > > p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < num_bin_x; i++ )
      for(int j =0; j<num_bin_y;j++)
        for(std::list<particle_t>::iterator particle_iter = p[i][j].begin();
            particle_iter != p[i][j].end(); particle_iter++)
            {
              fprintf( f, "%g %g\n", (*particle_iter).x, (*particle_iter).y );
            }

    // for( int i = 0; i < n; i++ )
    //     fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  MPI I/O routines
//
void J_MPI_save( FILE *f, int n, std::vector<std::vector<std::list<particle_t> > > &p, int rank, int region_num_bin_x, int region_num_bin_y)
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < region_num_bin_x; i++ )
      for(int j =0; j<region_num_bin_y;j++)
        for(std::list<particle_t>::iterator particle_iter = p[i][j].begin();
            particle_iter != p[i][j].end(); particle_iter++)
            {
              fprintf( f, "%d %g %g\n", rank, (*particle_iter).x, (*particle_iter).y );
            }

    // for( int i = 0; i < n; i++ )
    //     fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}
