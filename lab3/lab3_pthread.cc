#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>


unsigned long long ncpus;
unsigned long long pixels = 0;
unsigned long long r,k;
int part=0;
pthread_mutex_t  mutexsum;


void *cal_pixel(void *threadid){
	int* tid = (int*)threadid;
	unsigned long long local_pixel;

	
	unsigned long long partition = r/ncpus;
	unsigned long long r_sqr= r*r;
	

	if (r%ncpus != 0&& *tid<r%ncpus) partition++;
	unsigned long long offset = *tid >= r%ncpus ?  r%ncpus : 0;
	
	for(unsigned long long x =*tid*partition+offset ; x< *tid *partition+partition+offset  ; x++){
		local_pixel +=( ceil(sqrtl(r_sqr- x*x)));
		
	}
	local_pixel %= k;
	
	// for (unsigned long long x=*tid;x<r; x= x+ncpus){
	// 	local_pixel +=(ceil(sqrtl(r_sqr-x*x)));
	// 	local_pixel %= k;
	// }
	pthread_mutex_lock(&mutexsum);  
		pixels += local_pixel;
	pthread_mutex_unlock(&mutexsum);
	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	
	void *status;
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	 r = atoll(argv[1]);
	 k = atoll(argv[2]);
	 
	pthread_t threads[ncpus];
	pthread_mutex_init(&mutexsum,NULL);
	int rc;
    int ID[ncpus];
    int t;
    for (t = 0; t < ncpus; t++){
        ID[t] = t;
       
        rc = pthread_create(&threads[t], NULL,cal_pixel, (void*)&ID[t]);
    
    
    }
	

   

	for (t=0;t<  ncpus ;t++){
		pthread_join(threads[t],&status);
	}
	pthread_mutex_destroy(&mutexsum);

	printf("%llu\n", (4*pixels)%k );
	
	

	
    return 0;
	
}