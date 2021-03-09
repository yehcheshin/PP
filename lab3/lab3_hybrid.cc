#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	

	int rank,size;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	
	unsigned long long local_pixel=0;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	

	unsigned long long partition = (r/size) ;
	if (r%size != 0&& rank<r%size) partition++;
	unsigned long long offset = rank >= r%size ?  r%size : 0;
	unsigned long long r_sqr = r*r;
	#pragma omp parallel shared(local_pixel,partition) 
	{
		
		#pragma omp for schedule(static,1000) reduction(+:local_pixel) nowait
		for (unsigned long long x =rank*partition+offset ; x<rank*partition+partition+offset  ; x++) {
				local_pixel += ceil(sqrtl(r_sqr - x*x));
				
		}
		local_pixel %= k;
	}

	MPI_Reduce(&local_pixel,&pixels,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
	if(rank==0){
		printf("%llu\n",(4*pixels)%k);
	}
	
	MPI_Finalize();
	return 0;
}

