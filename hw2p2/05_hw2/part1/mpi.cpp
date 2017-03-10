#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <mpi.h>
#include <iostream>

#define probe_rank 0
#define probe_switch 1
#define debugWait 0


#include <vector>
#include <list>

int checkNeighProc(int proc_id, int neigh_local_id);

void clearSendRecvNumVars();
void populateSendVars(std::vector<std::vector<std::list<particle_t> > > &particles);
void sendRecvNumGhosts();
void resizeRecvVars();
void sendRecvGhosts();
void unpackRecvGhostsIntoLists();
std::vector<std::list<particle_t> > getGhostVec(std::vector<int> num, std::vector<particle_t> r_particles);

//variables to store send ghost particles info
std::vector<particle_t> s_TL, s_TR, s_BR, s_BL;
std::vector<particle_t> s_T, s_B, s_L, s_R;
std::vector<int> num_particle_s_T,num_particle_s_B,num_particle_s_L,num_particle_s_R;
std::vector<int> num_particle_s_TL,num_particle_s_TR,num_particle_s_BR,num_particle_s_BL;

//variables to store receive ghost particles info
std::vector<particle_t> r_TL, r_TR, r_BR, r_BL;
std::vector<particle_t> r_T, r_B, r_L, r_R;
std::vector<int> num_particle_r_T,num_particle_r_B,num_particle_r_L,num_particle_r_R;
std::vector<int> num_particle_r_TL,num_particle_r_TR,num_particle_r_BR,num_particle_r_BL;

//variables to store unpacked ghost particles
std::vector<std::list<particle_t> > ghosts_TL, ghosts_TR, ghosts_BR, ghosts_BL;
std::vector<std::list<particle_t> > ghosts_T, ghosts_R, ghosts_B, ghosts_L;

std::vector<std::vector<int> > myOrderedNeighs;

int region_num_bin_x,region_num_bin_y,num_regions,n_proc,rank,num_proc_x,num_proc_y;
double region_size_x,region_size_y,bin_size_x,bin_size_y;

MPI_Comm region_comm;
MPI_Datatype MPI_PARTICLE;
//
//  benchmarking program
//
int main( int argc, char **argv )
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    // std::cout<<"hey 1"<<std::endl;
    //
    //  process command line parameters
    //
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

    //
    //  set up MPI
    //
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // std::cout<<"my rank:"<<rank<<std::endl;

    //
    //  allocate generic resources
    //
    //FILE *fsave = savename && rank == 0 ? fopen( savename, "a" ) : NULL;
    FILE *fsave = savename ? fopen( savename, "a" ) : NULL;

    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    set_size( n );
    double size=J_get_size();

    // if ((rank==probe_rank)&&(probe_switch==1))
    //   std::cout<<"size: "<<size<<std::endl;

    //
    // Set up the dimensions, locations of regions that each processor is incharged of
    //
    num_proc_x = (int) sqrt(n_proc);
    int max_num_region_x_y = (int)floor(size/(J_get_cutoff()*1.01));

    if (max_num_region_x_y<num_proc_x)
    {
      num_proc_x = max_num_region_x_y;
      num_proc_y = max_num_region_x_y;
    }
    else
    {
      num_proc_y = n_proc/num_proc_x;
      if (max_num_region_x_y<num_proc_y)
      {
        num_proc_y = max_num_region_x_y;
      }
    }

      //  std::cout<<"num proc x"<<num_proc_x<<size<<std::endl;
      //  std::cout<<"num proc y"<<num_proc_y<<size<<std::endl;

    // std::cout<<"hey 1b "<<std::endl;

    num_regions = num_proc_x * num_proc_y;

    region_size_x = size/num_proc_x,
    region_size_y = size/num_proc_y;
    region_num_bin_x = (int)floor(region_size_x/(J_get_cutoff()*1.01));
    region_num_bin_y = (int)floor(region_size_y/(J_get_cutoff()*1.01));
    bin_size_x=region_size_x/region_num_bin_x;
    bin_size_y=region_size_y/region_num_bin_y;

    // std::cout<<"hey 1c "<<std::endl;
    //////std::cout<<"region_num_bin_x: "<< region_num_bin_x<<std::endl;
    //////std::cout<<"region_num_bin_y: "<< region_num_bin_y<<std::endl;

    std::vector<std::vector<double> > region_location;
    std::vector<std::vector<int> > region_indices;
    region_location.resize(num_regions);
    region_indices.resize(num_regions);

    for (int i=0; i<num_regions;++i)
    {
      region_location[i].push_back((i%num_proc_x)*region_size_x);
      region_location[i].push_back((i/num_proc_x)*region_size_y);
      region_indices[i].push_back(i%num_proc_x);
      region_indices[i].push_back(i/num_proc_x);
      // if ((rank==probe_rank)&&(probe_switch==1))
      // {
      //   std::cout<<region_location[i][0]<<std::endl;
      //   std::cout<<region_location[i][1]<<std::endl;
      //   std::cout<<region_indices[i][0]<<std::endl;
      //   std::cout<<region_indices[i][1]<<std::endl;
      //   std::cout<<std::endl;
      // }
    }
    //return 0;

    // std::cout<<"hey 1d "<<std::endl;
    //setting up vector to determine a processor's neighouring regions
    int neigh_proc;
    myOrderedNeighs.resize(num_regions);
    for (int i =0;i<num_regions;i++)
    {
      myOrderedNeighs[i].resize(8);
      myOrderedNeighs[i][0] =  checkNeighProc(i,0);
      myOrderedNeighs[i][1] =  checkNeighProc(i,1);
      myOrderedNeighs[i][2] =  checkNeighProc(i,2);
      myOrderedNeighs[i][3] =  checkNeighProc(i,3);
      myOrderedNeighs[i][4] =  checkNeighProc(i,4);
      myOrderedNeighs[i][5] =  checkNeighProc(i,5);
      myOrderedNeighs[i][6] =  checkNeighProc(i,6);
      myOrderedNeighs[i][7] =  checkNeighProc(i,7);

      // if ((rank==1)&&(probe_switch==1))
      // {
      //   std::cout<<"myOrderedNeighs: "<<      myOrderedNeighs[i][0]<<", "<<myOrderedNeighs[i][1]
      //                                 <<", "<<myOrderedNeighs[i][2]<<", "<<myOrderedNeighs[i][3]
      //                                 <<", "<<myOrderedNeighs[i][4]<<", "<<myOrderedNeighs[i][5]
      //                                 <<", "<<myOrderedNeighs[i][6]<<", "<<myOrderedNeighs[i][7]<<std::endl;
      // }

    }

    //return 0;

    ////particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    std::vector<std::vector<std::list<particle_t> > > J_all_particles, J_region_particles;
    J_region_particles.resize(region_num_bin_x);
    for (int i = 0; i<region_num_bin_x;i++)
      J_region_particles[i].resize(region_num_bin_y);

    //initializing all the particles using the same seed to ensure same initial condition throughout all processes
    J_MPI_init_particles(n, J_all_particles,region_num_bin_x*num_proc_x,region_num_bin_y*num_proc_y ,9999999);

    // std::cout<<"hey 1e "<<std::endl;
    // ////std::cout<<"rank: "<<rank<<",region_indices[rank][0]: "<< region_indices[rank][0]<<std::endl;
    // ////std::cout<<"rank: "<<rank<<",region_indices[rank][1]: "<< region_indices[rank][1]<<std::endl;
    // if (rank==0)
    // for(int i=0; i<(region_num_bin_x*num_proc_x);i++)
    //   for(int j=0; j<(region_num_bin_y*num_proc_y);j++)
    //   {
    //     ////std::cout<<"rank:"<<rank<<",size J all i,j: "<<i<<","<<j<<" | size:"<<J_all_particles[i][j].size()<<std::endl;
    //   }
    //
    // return 0;

    //copying the region that is relevant for the process
    if (rank<num_regions)
    for(int i=0; i<region_num_bin_x;i++)
      for(int j=0; j<region_num_bin_y;j++)
      {
        J_region_particles[i][j] = J_all_particles[region_indices[rank][0]*region_num_bin_x+i][region_indices[rank][1]*region_num_bin_y+j];
      }

// std::cout<<"hey 1f "<<std::endl;
    //
    // Preparing MPI datatype for sending particle across processes
    //
    const int    nItems=6;
    int          blocklengths[nItems] = {1,1,1,1,1,1};
    MPI_Datatype types[nItems] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint     offsets[nItems];

    offsets[0] = offsetof(particle_t, x);
    offsets[1] = offsetof(particle_t, y);
    offsets[2] = offsetof(particle_t, vx);
    offsets[3] = offsetof(particle_t, vy);
    offsets[4] = offsetof(particle_t, ax);
    offsets[5] = offsetof(particle_t, ay);

    MPI_Datatype MPI_PARTICLE_proto;
    MPI_Type_create_struct(nItems, blocklengths, offsets, types, &MPI_PARTICLE_proto);
    MPI_Type_commit(&MPI_PARTICLE_proto);

    // Resize the type so that its length matches the actual structure length

    // Get the constructed type lower bound and extent
    MPI_Aint lb, extent;
    MPI_Type_get_extent(MPI_PARTICLE_proto, &lb, &extent);

    // Get the actual distance between to vector elements
    // (this might not be the best way to do it - if so, substitute a better one)
    std::vector<particle_t> vec_Particle(2);
    extent = (char*)&vec_Particle[1] - (char*)&vec_Particle[0];

    // Create a resized type whose extent matches the actual distance
    MPI_Type_create_resized(MPI_PARTICLE_proto, lb, extent, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    //
    // Preparing MPI communicator for the working processes
    //
    // MPI_Group group_world, region_group;
    // int *members= new int[num_regions];
    // MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    //
    // for (int i=0; i<num_regions;i++)
    //   members[i] = i;
    //
    // MPI_Group_incl(group_world, num_regions, members, &region_group);
    // MPI_Comm_create(MPI_COMM_WORLD, region_group, &region_comm);

    int colour ,key, error ;
    if (rank<num_regions)
    {
      colour = 1;
      key = rank;
    }
    else
    {
      colour = MPI_UNDEFINED;
      key = MPI_UNDEFINED;
    }

    // std::cout<<"hey 1g "<<std::endl;
    error = MPI_Comm_split (MPI_COMM_WORLD, colour , key , &region_comm ) ;

    //
    //  set up the data partitioning across processors
    //
    // int particle_per_proc = (n + n_proc - 1) / n_proc;
    // int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    // for( int i = 0; i < n_proc+1; i++ )
    //     partition_offsets[i] = min( i * particle_per_proc, n );
    //
    // int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    // for( int i = 0; i < n_proc; i++ )
    //     partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

    //
    //  allocate storage for local partition
    //
    // int nlocal = partition_sizes[rank];
    // particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    // if( rank == 0 )
    //     init_particles( n, particles );
    // MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );


    //
    // Set up variables to send and receive number of ghost particles
    //
    num_particle_s_TL.resize(1);
    num_particle_s_TR.resize(1);
    num_particle_s_BL.resize(1);
    num_particle_s_BR.resize(1);
    num_particle_s_T.resize(region_num_bin_x);
    num_particle_s_B.resize(region_num_bin_x);
    num_particle_s_L.resize(region_num_bin_y);
    num_particle_s_R.resize(region_num_bin_y);

    num_particle_r_TL.resize(1);
    num_particle_r_TR.resize(1);
    num_particle_r_BL.resize(1);
    num_particle_r_BR.resize(1);
    num_particle_r_T.resize(region_num_bin_x);
    num_particle_r_B.resize(region_num_bin_x);
    num_particle_r_L.resize(region_num_bin_y);
    num_particle_r_R.resize(region_num_bin_y);

    //set up variables to unpack ghost particles into
    ghosts_TL.resize(1);
    ghosts_TR.resize(1);
    ghosts_BR.resize(1);
    ghosts_BL.resize(1);
    ghosts_T.resize(region_num_bin_x);
    ghosts_B.resize(region_num_bin_x);
    ghosts_R.resize(region_num_bin_y);
    ghosts_L.resize(region_num_bin_y);

    // std::cout<<"hey 2"<<std::endl;
    ////std::cout<<"size: "<<size<<std::endl;

    std::vector<std::vector<std::list<particle_t> > > intraBound_fallOut_particles; //to catch particles that have fallen out of their bins in a particular region
    intraBound_fallOut_particles.resize(region_num_bin_x);
    for (int i = 0; i<region_num_bin_x;i++)
    {
      intraBound_fallOut_particles[i].resize(region_num_bin_y);
    }
    std::vector<particle_t> extraBound_fallOut_particles;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    if (rank<num_regions) //only iterate if you are one of the working processes
    for( int step = 0; step < NSTEPS; step++ )
    {

      // std::cout<<rank<<": , step;"<<step<<std::endl;

      //
      //clear all the shits from previous iteration
      //

      //clear intraBound and extraBound vectors
      for (int i = 0; i<region_num_bin_x;i++)
        for (int j=0;j<region_num_bin_y;j++)
        intraBound_fallOut_particles[i][j].clear();
      extraBound_fallOut_particles.clear();
      //clear ghost particles
      ghosts_TL[0].clear();
      ghosts_TR[0].clear();
      ghosts_BR[0].clear();
      ghosts_BL[0].clear();
      for (int i=0;i<region_num_bin_x;i++)
      {
        ghosts_T[i].clear();
        ghosts_B[i].clear();
      }
      for (int j=0;j<region_num_bin_y;j++)
      {
        ghosts_R[j].clear();
        ghosts_L[j].clear();
      }

      // for (int i = 0; i<region_num_bin_x;i++)
      // {
      //   for (int j =0;j<region_num_bin_y;j++)
      //   {
      //     for(std::list<particle_t>::iterator particle_iter = J_region_particles[i][j].begin();
      //         particle_iter != J_region_particles[i][j].end(); particle_iter++)
      //         {
      //           ////std::cout<<"rank: "<<rank<<", particle x:"<<particle_iter->x<<std::endl;
      //           ////std::cout<<"rank: "<<rank<<", particle y:"<<particle_iter->y<<std::endl;
      //         }
      //
      //   }
      // }
      // return 0;
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        //
        // Preparing send and receive ghost zones (8 for each operation)
        //
        ////std::cout<<rank<<": hey 3"<<std::endl;
        clearSendRecvNumVars();
        ////std::cout<<rank<<": hey 3.5"<<std::endl;
        populateSendVars(J_region_particles);
        //return 0;
        // if ((rank==probe_rank)&&(probe_switch==1))
          // std::cout<<rank<<": hey 4"<<std::endl;
        sendRecvNumGhosts();

        // if ((rank==probe_rank)&&(probe_switch==1))
          // std::cout<<rank<<": hey 5"<<std::endl;
        //clearRecvVars();
        resizeRecvVars();

        // //
        // //  collect all global data locally (not good idea to do)
        // //
        // MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, region_comm );

        //
        // Send and receive ghost particles!
        //
        // if ((rank==probe_rank)&&(probe_switch==1))
        //   std::cout<<rank<<": hey 6"<<std::endl;
        sendRecvGhosts();

        //
        // Turn the linear vector of particles in the 8 ghost regions to vector of list of particles.
        //
        //if (rank==1)
        // if ((rank==probe_rank)&&(probe_switch==1))
        //   std::cout<<rank<<": hey 7"<<std::endl;
        unpackRecvGhostsIntoLists();

        // if ((rank==probe_rank)&&(probe_switch==1))
        // {
        // std::cout<<rank<<": hey 7b"<<std::endl;
        // for (int i =0; i<num_particle_r_R.size();i++)
        // {
        //   std::cout <<"R: "<< num_particle_r_R[i]<<std::endl;
        //   std::cout <<"L: "<<num_particle_r_L[i]<<std::endl;
        // }
        // for (int i =0; i<num_particle_r_T.size();i++)
        // {
        //   std::cout <<"T: "<< num_particle_r_T[i]<<std::endl;
        //   std::cout <<"B: "<<num_particle_r_B[i]<<std::endl;
        // }
        // std::cout <<"TL: "<< num_particle_r_TL[0]<<std::endl;
        // std::cout <<"TR: "<< num_particle_r_TR[0]<<std::endl;
        // std::cout <<"BL: "<< num_particle_r_BL[0]<<std::endl;
        // std::cout <<"BR: "<< num_particle_r_BR[0]<<std::endl;
        // }

        //if ((rank==probe_rank)&&(probe_switch==1))
      //  {
          // int total_particles =0;
          // for( int i = 0; i < region_num_bin_x; i++ )
          //   for(int j =0; j<region_num_bin_y;j++)
          //     total_particles += J_region_particles[i][j].size();
          // std::cout<<rank<<": Number of particles: "<<total_particles<<std::endl;
        //}


        //std::cout<<rank<<": hey 8"<<std::endl;
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            //save( fsave, n, particles );
            J_MPI_save( fsave, n, J_region_particles, rank, region_num_bin_x, region_num_bin_y );

        // //
        // //  compute all forces
        // //
        // for( int i = 0; i < nlocal; i++ )
        // {
        //     local[i].ax = local[i].ay = 0;
        //     for (int j = 0; j < n; j++ )
        //         apply_force( local[i], particles[j], &dmin, &davg, &navg );
        // }

        //
        //  compute forces
        //

        int neigh_x,neigh_y;
        for( int i = 0; i < region_num_bin_x; i++ )
          for(int j =0; j<region_num_bin_y;j++)
            for(std::list<particle_t>::iterator particle_iter = J_region_particles[i][j].begin();
                particle_iter != J_region_particles[i][j].end(); particle_iter++)
                {
                  (*particle_iter).ax =0;
                  (*particle_iter).ay = 0;
                  for (int del_i = -1; del_i<=1; del_i++)
                    for(int del_j=-1;del_j<=1;del_j++)
                    {
                      neigh_x = i+del_i;
                      neigh_y = j+del_j;
                      if ((neigh_x>=0)&&(neigh_x<region_num_bin_x) && (neigh_y>=0)&&(neigh_y<region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = J_region_particles[neigh_x][neigh_y].begin();
                            neigh_iter != J_region_particles[neigh_x][neigh_y].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x>=0)&&(neigh_x<region_num_bin_x) && (neigh_y==-1))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_T[neigh_x].begin();
                            neigh_iter != ghosts_T[neigh_x].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x>=0)&&(neigh_x<region_num_bin_x) && (neigh_y==region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_B[neigh_x].begin();
                            neigh_iter != ghosts_B[neigh_x].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==-1)&& (neigh_y>=0)&&(neigh_y<region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_L[neigh_y].begin();
                            neigh_iter != ghosts_L[neigh_y].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==region_num_bin_x)&& (neigh_y>=0)&&(neigh_y<region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_R[neigh_y].begin();
                            neigh_iter != ghosts_R[neigh_y].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==-1) && (neigh_y==-1))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_TL[0].begin();
                            neigh_iter != ghosts_TL[0].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==-1) && (neigh_y==region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_BL[0].begin();
                            neigh_iter != ghosts_BL[0].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==region_num_bin_x)&& (neigh_y==-1))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_TR[0].begin();
                            neigh_iter != ghosts_TR[0].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);

                      else if ((neigh_x==region_num_bin_x)&&(neigh_y==region_num_bin_y))
                        for(std::list<particle_t>::iterator neigh_iter = ghosts_BR[0].begin();
                            neigh_iter != ghosts_BR[0].end(); neigh_iter++)
                              apply_force( *particle_iter, *neigh_iter,&dmin,&davg,&navg);
                    }
                }
//std::cout<<rank<<":hey 9"<<std::endl;
        if( find_option( argc, argv, "-no" ) == -1 )
        {

          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,region_comm);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,region_comm);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,region_comm);


          if (rank == 0){
            //
            // Computing statistical data
            //

            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        // for( int i = 0; i < nlocal; i++ )
        //     move( local[i] );

        // ////std::cout<<"before extrabound"<<extraBound_fallOut_particles.size()<<std::endl;
        // int temp_intra_size=0;
        // for(int i=0;i<region_num_bin_x;i++)
        //   for(int j=0;j<region_num_bin_y;j++)
        //     temp_intra_size+=intraBound_fallOut_particles[i][j].size();
        //
        // ////std::cout<<"before intrabound"<<temp_intra_size<<std::endl;

        int new_bin_x,new_bin_y;
        for( int i = 0; i < region_num_bin_x; i++ )
          for(int j =0; j<region_num_bin_y;j++)
            for(std::list<particle_t>::iterator particle_iter = J_region_particles[i][j].begin();
                particle_iter != J_region_particles[i][j].end(); particle_iter++)
                {
                  move( *particle_iter );
                  ////std::cout<<"rank"<<rank<<", particle x: "<<particle_iter->x<<std::endl;
                  ////std::cout<<"rank"<<rank<<", particle y: "<<particle_iter->y<<std::endl;
                  new_bin_x = (int)floor((particle_iter->x-region_location[rank][0])/(bin_size_x));
                  new_bin_y = (int)floor((particle_iter->y-region_location[rank][1])/(bin_size_y));
                  if((new_bin_x != i)||(new_bin_y != j))
                  {
                    std::list<particle_t>::iterator temp_iter = particle_iter;
                    --particle_iter;
                    //omp_set_lock(lock_J_particles[i][j]);
                   //if (()||())
                    if ((new_bin_x <0)||(new_bin_x >=region_num_bin_x)||(new_bin_y <0)||(new_bin_y >=region_num_bin_y))
                    {
                      extraBound_fallOut_particles.push_back(*temp_iter);
                      J_region_particles[i][j].erase (temp_iter);
                    }
                    else
                    {
                      intraBound_fallOut_particles[new_bin_x][new_bin_y].splice (intraBound_fallOut_particles[new_bin_x][new_bin_y].begin(),
                      J_region_particles[i][j],temp_iter);
                    }
                  }
                }
                //////std::cout<<"size:"<<size<<std::endl;
                //////std::cout<<"region_location[rank][0]:"<<region_location[rank][0]<<std::endl;
                //////std::cout<<"region_location[rank][1]:"<<region_location[rank][1]<<std::endl;


                ////std::cout<<"after extrabound"<<extraBound_fallOut_particles.size()<<std::endl;
                int temp_intra_size=0;
                for(int i=0;i<region_num_bin_x;i++)
                  for(int j=0;j<region_num_bin_y;j++)
                    temp_intra_size+=intraBound_fallOut_particles[i][j].size();

                ////std::cout<<"after intrabound"<<temp_intra_size<<std::endl;

                //
                //  add fallen particles into their new bins within the region
                //
                for( int i = 0; i < region_num_bin_x; i++ )
                  for(int j =0; j<region_num_bin_y;j++)
                    J_region_particles[i][j].splice (J_region_particles[i][j].begin(),intraBound_fallOut_particles[i][j]);

                //
                // Dealing with particles that have fallen out of a region and requires communication
                //

                int *r_num_extraBoundary_particles=new int[num_regions],
                    *displacements= new int [num_regions];
                int size_temp = extraBound_fallOut_particles.size();
                ////std::cout<<"num_regions:"<<num_regions<<std::endl;
                // for(int i=0;i<extraBound_fallOut_particles.size();i++)
                // {
                //   std::cout<<rank<<": "<<": particle_x: "<<extraBound_fallOut_particles[i].x<<std::endl;
                //   std::cout<<rank<<": particle_y: "<<extraBound_fallOut_particles[i].y<<std::endl;
                //   std::cout<<rank<<": particle_vx: "<<extraBound_fallOut_particles[i].vx<<std::endl;
                //   std::cout<<rank<<": particle_vy: "<<extraBound_fallOut_particles[i].vy<<std::endl;
                //   std::cout<<rank<<": particle_ax: "<<extraBound_fallOut_particles[i].ax<<std::endl;
                //   std::cout<<rank<<": particle_ay: "<<extraBound_fallOut_particles[i].ay<<std::endl;
                // }

                //MPI_Barrier(region_comm);
                //get the number of particles that has dropped out of each region
                MPI_Allgather(&size_temp, 1, MPI_INT, r_num_extraBoundary_particles, 1,MPI_INT, region_comm);
                // for(int i=0;i<num_regions;i++)
                // {
                //   std::cout<<"region: "<<i<<", number of extrabound particles: "<<r_num_extraBoundary_particles[i]<<std::endl;
                // }
                int totalNumExtraBound=0;
                for (int i =0;i<num_regions;i++)
                {
                  //////std::cout<<"r_num_extraBoundary_particles[i]:"<<r_num_extraBoundary_particles[i]<<std::endl;
                  totalNumExtraBound += r_num_extraBoundary_particles[i];
                }

                std::vector<particle_t> r_extra_particles;
                r_extra_particles.resize(totalNumExtraBound);

                // std::vector<double> r_extra_particles;
                // r_extra_particles.resize(6*totalNumExtraBound);

                MPI_Aint disp;
                MPI_Type_extent(MPI_PARTICLE, &disp);

                // if ((rank==probe_rank)&&(probe_switch))
                // std::cout<<"extent of mpi_particle: "<<disp<<std::endl;

                displacements[0] = static_cast<MPI_Aint>(0);
                //std::cout<<rank<<": r_num_extraBoundary_particles: "<<0<<": "<<r_num_extraBoundary_particles[0]<<std::endl;
                //std::cout<<rank<<": displacement "<<0<<": "<<displacements[0]<<std::endl;

                for (int i=1;i<num_regions;i++)
                {
                  displacements[i] = displacements[i-1] + r_num_extraBoundary_particles[i-1];
                  //std::cout<<rank<<": r_num_extraBoundary_particles: "<<i<<": "<<r_num_extraBoundary_particles[i]<<std::endl;
                  // if ((rank==probe_rank)&&(probe_switch))
                  //   std::cout<<rank<<": displacement "<<i<<": "<<displacements[i]<<std::endl;
                }
                //if ((rank==probe_rank)&&(probe_switch))
                // {
                //   std::cout<<rank<<": r_num_extraBoundary_particles: ";
                //   for (int i=0;i<num_regions;i++)
                //     std::cout<<r_num_extraBoundary_particles[i]<<", ";
                //   std::cout<<std::endl;
                // }

                //std::cout<<rank<<": hey 11"<<std::endl;
                ////std::cout<<extraBound_fallOut_particles.size();
                // get all the particles that have dropped out of the regions to all processes
                MPI_Allgatherv(&extraBound_fallOut_particles.front(), extraBound_fallOut_particles.size(), MPI_PARTICLE, &r_extra_particles.front(), r_num_extraBoundary_particles,
                              displacements, MPI_PARTICLE, region_comm);

                //MPI_Barrier(region_comm);
//std::cout<<rank<<": hey 12"<<std::endl;

                //Putting relevant extrabound particles into the bin of the relevant region

                //std::cout<<r_extra_particles;
                int bin_x,bin_y;
                for(std::vector<particle_t>::iterator particle_iter = r_extra_particles.begin();
                    particle_iter != r_extra_particles.end(); particle_iter++)
                    {
                      // if ((rank==0)&&(probe_switch==1))
                      // if ((probe_switch==1))
                      // {
                      //   std::cout<<rank<<": hey 12a"<<std::endl;
                      //   std::cout<<rank<<": particle_x: "<<particle_iter->x<<std::endl;
                      //   std::cout<<rank<<": particle_y: "<<particle_iter->y<<std::endl;
                      //   std::cout<<rank<<": particle_vx: "<<particle_iter->vx<<std::endl;
                      //   std::cout<<rank<<": particle_vy: "<<particle_iter->vy<<std::endl;
                      //   std::cout<<rank<<": particle_ax: "<<particle_iter->ax<<std::endl;
                      //   std::cout<<rank<<": particle_ay: "<<particle_iter->ay<<std::endl;
                      // }
                      bin_x = (int)floor((particle_iter->x-region_location[rank][0])/(bin_size_x));
                      bin_y = (int)floor((particle_iter->y-region_location[rank][1])/(bin_size_y));
                      //std::cout<<rank<<": hey 12b"<<std::endl;
                      if ((bin_x>=0)&&(bin_x<region_num_bin_x) && (bin_y>=0)&&(bin_y<region_num_bin_y))
                      {
                        //std::cout<<rank<<": hey 12c"<<std::endl;
                        J_region_particles[bin_x][bin_y].push_front(*particle_iter);
                        //std::cout<<rank<<": hey 12d"<<std::endl;
                      }
                    }

                // int bin_x,bin_y;
                // for(std::vector<double>::iterator particle_iter = r_extra_particles.begin();
                //     particle_iter != r_extra_particles.end(); particle_iter++)
                //     {
                //       if ((rank==1)&&(probe_switch==1))
                //       if ((probe_switch==1))
                //       {
                //         std::cout<<rank<<": hey 12a"<<std::endl;
                //         std::cout<<rank<<": particle_x: "<<(*particle_iter)<<std::endl;
                //         std::cout<<rank<<": particle_y: "<<(*(++particle_iter))<<std::endl;
                //         std::cout<<rank<<": particle_vx: "<<(*(++particle_iter))<<std::endl;
                //         std::cout<<rank<<": particle_vy: "<<(*(++particle_iter))<<std::endl;
                //         std::cout<<rank<<": particle_ax: "<<(*(++particle_iter))<<std::endl;
                //         std::cout<<rank<<": particle_ay: "<<(*(++particle_iter))<<std::endl;
                //       }
                //       particle_iter -= 6;
                //       bin_x = (int)floor(((*particle_iter)-region_location[rank][0])/(bin_size_x));
                //       bin_y = (int)floor(((*(++particle_iter))-region_location[rank][1])/(bin_size_y));
                //       //std::cout<<rank<<": hey 12b"<<std::endl;
                //       particle_iter -=2;
                //       if ((bin_x>=0)&&(bin_x<region_num_bin_x) && (bin_y>=0)&&(bin_y<region_num_bin_y))
                //       {
                //         //std::cout<<rank<<": hey 12c"<<std::endl;
                //         particle_t temp = {(*particle_iter),(*(++particle_iter))};
                //         J_region_particles[bin_x][bin_y].push_front(temp);
                //         //std::cout<<rank<<": hey 12d"<<std::endl;
                //         particle_iter -=2;
                //       }
                //       particle_iter += 6;
                //     }

                  // if ((rank==probe_rank)&&(probe_switch==1))
                  //   std::cout<<rank<<": hey 13"<<std::endl;

                  //delete all the shits
                  delete[] r_num_extraBoundary_particles;
                  delete[] displacements;

    }


    // if ((rank==probe_rank)&&(probe_switch==1))
    //   std::cout<<rank<<": hey 14m"<<std::endl;

    simulation_time = read_timer() - simulation_time;

    if (rank == 0) {
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }

    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    //free( partition_offsets );
    //free( partition_sizes );
    //free( local );
    //free( particles );
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}

int checkNeighProc(int proc_id,int neigh_local_id)
{
  int neigh_residue,neigh_proc_id,delta_x;
  int my_residue = proc_id%num_proc_x;
  switch(neigh_local_id)
  {
    case 0:
    neigh_proc_id = proc_id-1-num_proc_x;
    delta_x=-1;
    break;
    case 1:
    neigh_proc_id = proc_id-num_proc_x;
    delta_x=0;
    break;
    case 2:
    neigh_proc_id = proc_id+1-num_proc_x;
    delta_x=1;
    break;
    case 3:
    neigh_proc_id = proc_id-1;
    delta_x=-1;
    break;
    case 4:
    neigh_proc_id = proc_id+1;
    delta_x=1;
    break;
    case 5:
    neigh_proc_id = proc_id-1+num_proc_x;
    delta_x=-1;
    break;
    case 6:
    neigh_proc_id = proc_id+num_proc_x;
    delta_x=0;
    break;
    case 7:
    neigh_proc_id = proc_id+1+num_proc_x;
    delta_x=1;
    break;
  }

  neigh_residue = neigh_proc_id % num_proc_x;
  if ((neigh_proc_id<0)||(neigh_proc_id>=num_regions)||((my_residue+delta_x)!=neigh_residue))
    return -1;
  else
    return neigh_proc_id;

}

void clearSendRecvNumVars()
{
  //clear send and receive num vectors
  num_particle_s_TL[0]=0;
  num_particle_s_TR[0]=0;
  num_particle_s_BL[0]=0;
  num_particle_s_BR[0]=0;
  num_particle_r_TL[0]=0;
  num_particle_r_TR[0]=0;
  num_particle_r_BL[0]=0;
  num_particle_r_BR[0]=0;
  for (int i=0;i<region_num_bin_x;i++)
  {
    num_particle_s_T[i]=0;
    num_particle_s_B[i]=0;
    num_particle_r_T[i]=0;
    num_particle_r_B[i]=0;
  }
  for (int j=0;j<region_num_bin_y;j++)
  {
    num_particle_s_L[j]=0;
    num_particle_s_R[j]=0;
    num_particle_r_L[j]=0;
    num_particle_r_R[j]=0;
  }
  //clear send vars
  s_TL.clear();
  s_TR.clear();
  s_BL.clear();
  s_BR.clear();
  s_T.clear();
  s_B.clear();
  s_L.clear();
  s_R.clear();
  //clear receive vars
  r_TL.clear();
  r_TR.clear();
  r_BL.clear();
  r_BR.clear();
  r_T.clear();
  r_B.clear();
  r_L.clear();
  r_R.clear();
}

void resizeRecvVars()
{
  int sum_T=0,sum_B=0,sum_L=0,sum_R=0;
  for (int i=0;i<region_num_bin_x;i++)
  {
    sum_T += num_particle_r_T[i];
    sum_B += num_particle_r_B[i];
    //std::cout<<num_particle_r_T[i]<<std::endl;
  }
  for (int j=0;j<region_num_bin_y;j++)
  {
    sum_L += num_particle_r_L[j];
    sum_R += num_particle_r_R[j];
  }
  //std::cout<<"i'm here"<<std::endl;
  //std::cout<<"sum_T: "<<sum_T<<std::endl;
  r_T.resize(sum_T);
  //std::cout<<"i'm here"<<std::endl;
  r_B.resize(sum_B);
  r_L.resize(sum_L);
  r_R.resize(sum_R);
  //std::cout<<"i'm here"<<std::endl;
  r_TL.resize(num_particle_r_TL[0]);
  //std::cout<<"i'm here"<<std::endl;
  r_TR.resize(num_particle_r_TR[0]);
  r_BR.resize(num_particle_r_BR[0]);
  r_BL.resize(num_particle_r_BL[0]);
  //std::cout<<"i'm here"<<std::endl;
}

void populateSendVars(std::vector<std::vector<std::list<particle_t> > > &particles)
{
  ////std::cout<<"hey 3a"<<std::endl;
  //s_TL.resize(particles[0][0].size());
  s_TL.insert(s_TL.begin(),particles[0][0].begin(), particles[0][0].end());
  num_particle_s_TL[0]=particles[0][0].size();
  ////std::cout<<"hey 3c"<<std::endl;
  //s_TR.resize(particles[region_num_bin_x-1][0].size());
  s_TR.insert(s_TR.begin(),particles[region_num_bin_x-1][0].begin(), particles[region_num_bin_x-1][0].end());
  num_particle_s_TR[0]=particles[region_num_bin_x-1][0].size();
////std::cout<<"hey 3d"<<std::endl;
  //s_BL.resize(particles[0][(region_num_bin_y-1)].size());
  s_BL.insert(s_BL.begin(),particles[0][(region_num_bin_y-1)].begin(), particles[0][(region_num_bin_y-1)].end());
  num_particle_s_BL[0]=particles[0][(region_num_bin_y-1)].size();
////std::cout<<"hey 3e"<<std::endl;
  //s_BR.resize(particles[region_num_bin_x-1][(region_num_bin_y-1)].size());
  s_BR.insert(s_BR.begin(),particles[region_num_bin_x-1][(region_num_bin_y-1)].begin(), particles[region_num_bin_x-1][(region_num_bin_y-1)].end());
  num_particle_s_BR[0]=particles[region_num_bin_x-1][(region_num_bin_y-1)].size();
  ////std::cout<<"hey 3f"<<std::endl;
  for (int i=0;i<region_num_bin_x;i++)
  {
    s_T.insert(s_T.end(),particles[i][0].begin(), particles[i][0].end());
    num_particle_s_T[i]=particles[i][0].size();

    s_B.insert(s_B.end(),particles[i][(region_num_bin_y-1)].begin(), particles[i][(region_num_bin_y-1)].end());
    num_particle_s_B[i]=particles[i][(region_num_bin_y-1)].size();

  }
  for (int j=0;j<region_num_bin_y;j++)
  {
    s_L.insert(s_L.end(),particles[0][j].begin(), particles[0][j].end());
    num_particle_s_L[j]=particles[0][j].size();

    s_R.insert(s_R.end(),particles[region_num_bin_x-1][j].begin(), particles[region_num_bin_x-1][j].end());
    num_particle_s_R[j]=particles[region_num_bin_x-1][j].size();

  }
}

void sendRecvNumGhosts()
{
  //std::vector<std::vector<int> > myOrderedNeighs;

  std::vector<int> neigh = myOrderedNeighs[rank];
  MPI_Request r[16];

  //std::cout<<"num region x: "<<region_num_bin_x<<std::mendl;

  int counter = -1;
  for(int i=0;i<8;i++)
  {
    if (neigh[i]!=-1) //test for boundary cases, -1 was used to encode these
    {
      counter++;
      switch(i)
      {
        case 0:
          ////std::cout<<"i'm here"<<std::endl;
          MPI_Isend(&num_particle_s_TL[0], 1, MPI_INT,
            neigh[i], 1, region_comm, &(r[counter]));
            break;
            ////std::cout<<"i'm here"<<std::endl;
        case 1:
        ////std::cout<<"i'm here"<<std::endl;
          MPI_Isend(&num_particle_s_T[0], region_num_bin_x, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 2:
          MPI_Isend(&num_particle_s_TR[0], 1, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 3:
          MPI_Isend(&num_particle_s_L[0], region_num_bin_y, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 4:
          MPI_Isend(&num_particle_s_R[0], region_num_bin_y, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 5:
          MPI_Isend(&num_particle_s_BL[0], 1, MPI_INT,
                  neigh[i], 1, region_comm,&(r[counter]));
                  break;
        case 6:
          MPI_Isend(&num_particle_s_B[0], region_num_bin_x, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 7:
          MPI_Isend(&num_particle_s_BR[0], 1, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;

      }

    }
  }

  //MPI_Request *r= new MPI_Request[counter];
  //MPI_Status status[counter];
  //counter =-1;
  for(int i=0;i<8;i++)
  {
    if (neigh[i]!=-1) //test for boundary cases, -1 was used to encode these
    {
      counter++;
      switch(i)
      {
        case 0:
          MPI_Irecv(&num_particle_r_TL[0], 1, MPI_INT,
                    neigh[i], 1, region_comm, &(r[counter]));
                    break;
        case 1:
          MPI_Irecv(&num_particle_r_T[0], region_num_bin_x, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 2:
          MPI_Irecv(&num_particle_r_TR[0], 1, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 3:
          MPI_Irecv(&num_particle_r_L[0], region_num_bin_y, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 4:
          MPI_Irecv(&num_particle_r_R[0], region_num_bin_y, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 5:
          MPI_Irecv(&num_particle_r_BL[0], 1, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 6:
          MPI_Irecv(&num_particle_r_B[0], region_num_bin_x, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;
        case 7:
          MPI_Irecv(&num_particle_r_BR[0], 1, MPI_INT,
                  neigh[i], 1, region_comm, &(r[counter]));
                  break;

      }

    }
  }

      MPI_Waitall(counter+1, r, MPI_STATUSES_IGNORE);
    //delete[] r;
  }

void sendRecvGhosts()
{
  std::vector<int> neigh = myOrderedNeighs[rank];
  MPI_Request request;

  int counter = 0;
  for(int i=0;i<8;i++)
    if (neigh[i]!=-1) //test for extra-boundary cases, -1 was used to encode these
    {
      counter++;
      switch(i)
      {
        case 0:
          MPI_Isend(&s_TL[0], s_TL.size(), MPI_PARTICLE,
                    neigh[i], 2, region_comm, &request);
                    break;
        case 1:
          MPI_Isend(&s_T[0], s_T.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 2:
          MPI_Isend(&s_TR[0], s_TR.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 3:
          MPI_Isend(&s_L[0], s_L.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 4:
          MPI_Isend(&s_R[0], s_R.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 5:
          MPI_Isend(&s_BL[0], s_BL.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 6:
          MPI_Isend(&s_B[0], s_B.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
        case 7:
          MPI_Isend(&s_BR[0], s_BR.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &request);
                  break;
      }

    }

    MPI_Request *r= new MPI_Request[counter];
  //MPI_Status status[counter];

  counter =-1;
  for(int i=0;i<8;i++)
    if (neigh[i]!=-1) //test for extra-boundary cases, -1 was used to encode these
    {
      counter++;
      switch(i)
      {
        case 0:
          MPI_Irecv(&r_TL[0], r_TL.size(), MPI_PARTICLE,
                    neigh[i], 2, region_comm, &(r[counter]));
                    break;
        case 1:
          MPI_Irecv(&r_T[0], r_T.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 2:
          MPI_Irecv(&r_TR[0], r_TR.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 3:
          MPI_Irecv(&r_L[0], r_L.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 4:
          MPI_Irecv(&r_R[0], r_R.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 5:
          MPI_Irecv(&r_BL[0], r_BL.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 6:
          MPI_Irecv(&r_B[0], r_B.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
        case 7:
          MPI_Irecv(&r_BR[0], r_BR.size(), MPI_PARTICLE,
                  neigh[i], 2, region_comm, &(r[counter]));
                  break;
      }

    }
      MPI_Waitall(counter+1, r, MPI_STATUSES_IGNORE);
}

void unpackRecvGhostsIntoLists()
{
  ghosts_TL= getGhostVec(num_particle_r_TL,r_TL);
  ghosts_TR= getGhostVec(num_particle_r_TR,r_TR);
  ghosts_BR= getGhostVec(num_particle_r_BR,r_BR);
  ghosts_BL= getGhostVec(num_particle_r_BL,r_BL);

  ghosts_T= getGhostVec(num_particle_r_T,r_T);
  ghosts_R= getGhostVec(num_particle_r_R,r_R);
  ghosts_B= getGhostVec(num_particle_r_B,r_B);
  ghosts_L= getGhostVec(num_particle_r_L,r_L);
}

std::vector<std::list<particle_t> > getGhostVec(std::vector<int> num, std::vector<particle_t> r_particles)
{
  std::vector<std::list<particle_t> > vec;
  vec.resize(num.size());

  std::vector<particle_t>::iterator it = r_particles.begin();

  for (int i=0;i<num.size();i++)
  {
    std::copy( it, it+num[i], std::back_inserter( vec[i] ) );
    it += num[i];
  }

  return vec;
}
