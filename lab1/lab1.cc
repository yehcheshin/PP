#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    //time test
    // struct timespec start,end,temp;
    // double time_used;
    // clock_gettime(CLOCK_MONOTONIC,&start);
    
    
    int rank,size;
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    
    unsigned long long local_pixel=0;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    unsigned long long partition = (r/size) ;
    if (!(rank==size-1)){
        for (unsigned long long x =rank*partition ; x<rank*partition+partition  ; x++) {
            //printf("process:%d X:%llu\n",rank,x);
            local_pixel += ceil(sqrtl(r*r - x*x));
            local_pixel %= k;
        }
    }
    else{
        
        for (unsigned long long x =rank*partition ; x<rank*partition+partition + r%size ; x++) {
            //printf("process:%d X:%llu\n",rank,x);
            //if(x%size!=rank)continue;
            local_pixel += ceil(sqrtl(r*r - x*x));
            local_pixel %= k;
        }
    }
    MPI_Reduce(&local_pixel,&pixels,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("%llu\n",(4*pixels)%k);
    }
    MPI_Finalize();
    return 0;
    
    
    
    // clock_gettime(CLOCK_MONOTONIC,&end);
    // if((end.tv_nsec -start.tv_nsec)<0){
    //     temp.tv_sec = end.tv_sec -start.tv_nsec -1;
    //     temp.tv_nsec = 1000000000 + end.tv_nsec -start.tv_nsec;
    // }else{
    //     temp.tv_sec = end.tv_sec -start.tv_sec;
    //     temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // time_used = temp.tv_sec +(double)temp.tv_nsec / 1000000000.0;
    // printf("%f second \n",time_used );
    
    
}
