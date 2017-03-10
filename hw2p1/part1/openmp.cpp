#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <iostream>


#include <vector>
#include <list>

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    int navg,nabsavg=0,numthreads;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    set_size( n );
    double size=J_get_size();
    int num_bin_x = (int)floor(J_get_size()/(J_get_cutoff()*1.00000001));
    int num_bin_y = (int)floor(J_get_size()/(J_get_cutoff()*1.00000001));

    ////particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    std::vector<std::vector<std::list<particle_t> > > J_particles;
    std::vector<std::vector<std::list<particle_t> > > fallOut_particles; //to catch particles that have fallen out of their bins
    fallOut_particles.resize(num_bin_x);
    for (int i = 0; i<num_bin_x;i++)
      fallOut_particles[i].resize(num_bin_y);

    // std::vector<std::vector<omp_lock_t > > lock_J_particles;
    // std::vector<std::vector<omp_lock_t > > lock_fallOut_particles;
    // lock_J_particles.resize(num_bin_x);
    // lock_fallOut_particles.resize(num_bin_x);
    // for (int i = 0; i<num_bin_x;i++)
    // {
    //   lock_J_particles[i].resize(num_bin_y);
    //   lock_fallOut_particles[i].resize(num_bin_y);
    //   for (int j=0; j<num_bin_y;j++)
    //   {
    //     omp_init_lock(&lock_J_particles[i][j]);
    //     omp_init_lock(&lock_fallOut_particles[i][j]);
    //   }
    // }

    ////init_particles( n, particles );
    J_init_particles(n, J_particles );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    #pragma omp parallel private(dmin)
    {
      numthreads = omp_get_num_threads();

      for( int step = 0; step < NSTEPS; step++ )
      {
  	     navg = 0;
          davg = 0.0;
  	       dmin = 1.0;
          //
          //  compute forces
          //

          int neigh_x,neigh_y;
          #pragma omp for reduction (+:navg) reduction(+:davg) collapse(2)
          for( int i = 0; i < num_bin_x; i++ )
            for(int j =0; j<num_bin_y;j++)
              for(std::list<particle_t>::iterator particle_iter = J_particles[i][j].begin();
                  particle_iter != J_particles[i][j].end(); particle_iter++)
                  {
                    (*particle_iter).ax = (*particle_iter).ay = 0;
                    for (int del_i = -1; del_i<=1; del_i++)
                      for(int del_j=-1;del_j<=1;del_j++)
                      {
                        neigh_x = i+del_i;
                        neigh_y = j+del_j;
                        if ((neigh_x>=0)&&(neigh_x<num_bin_x) && (neigh_y>=0)&&(neigh_y<num_bin_y))
                          for(std::list<particle_t>::iterator neigh_iter = J_particles[neigh_x][neigh_y].begin();
                              neigh_iter != J_particles[neigh_x][neigh_y].end(); neigh_iter++)
                                apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);
                      }
                  }

          //
          //  move particles and extract particles that have fallen out of their bins into a new variable
          //
          int new_bin_x,new_bin_y;
          #pragma omp for collapse(2)
          for( int i = 0; i < num_bin_x; i++ )
            for(int j =0; j<num_bin_y;j++)
              for(std::list<particle_t>::iterator particle_iter = J_particles[i][j].begin();
                  particle_iter != J_particles[i][j].end(); particle_iter++)
                  {
                    move( *particle_iter );
                    new_bin_x = (int)((particle_iter->x)/(size/num_bin_x));
                    new_bin_y = (int)((particle_iter->y)/(size/num_bin_y));
                    if((new_bin_x != i)||(new_bin_y != j))
                    {
                      std::list<particle_t>::iterator temp_iter = particle_iter;
                      --particle_iter;
                      //omp_set_lock(lock_J_particles[i][j]);
                      fallOut_particles[new_bin_x][new_bin_y].splice (fallOut_particles[new_bin_x][new_bin_y].begin(),
                      J_particles[i][j],temp_iter);
                      //omp_unset_lock(lock_J_particles[i][j]);
                    }
                  }

          //
          //  add fallen particles into their new bins
          //
          #pragma omp for collapse(2)
          for( int i = 0; i < num_bin_x; i++ )
            for(int j =0; j<num_bin_y;j++)
              J_particles[i][j].splice (J_particles[i][j].begin(),fallOut_particles[i][j]);


          if( find_option( argc, argv, "-no" ) == -1 )
          {
            //
            // Computing statistical data
            //
            #pragma omp master
            if (navg) {
              absavg +=  davg/navg;
              nabsavg++;
            }

            #pragma omp critical
            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //
            #pragma omp master
            if( fsave && (step%SAVEFREQ) == 0 )
                //save( fsave, n, particles );
                J_save( fsave, n, J_particles );
          }
      }
  }
    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    //
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    //free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
