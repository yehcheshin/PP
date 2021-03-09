#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int ncpus;
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 2;
	}
	
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	
	unsigned long long x = 0;
	unsigned long long r_sqr = r*r;

	
	#pragma omp parallel  shared(pixels)
	{	
	
		#pragma omp for schedule(static,10000) reduction(+:pixels) 
		for (unsigned long long x = 0; x < r; x++) {
			
			unsigned long long y = ceil(sqrtl(r_sqr - x*x));
			
			pixels += y;
			
		}
		pixels %= k;
		
	}
	printf("%llu\n", (4 * pixels) % k);
}
