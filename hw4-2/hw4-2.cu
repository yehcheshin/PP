#include    <stdio.h>
#include    <stdlib.h>
#include    <cuda_runtime.h>
#include    <omp.h>
#define BLOCK_SIZE 32
#define INF 1073741823
int V;
int E;
int *dist;
int *dev_dist[2];
void input_file(char*);
void write_file(char*);
void block_FW();
int ceil(int,int);

inline __device__ int Min(int temp,int d_temp){
    return min(temp,d_temp);
}
inline __device__ int check_bound(int a,int b,int V){
    if(a<V&&b<V)return 1;
    else return 0;
}
__global__ void phase1(int *,int ,int );
__global__ void phase2(int * ,int ,int );
__global__ void phase3(int * ,int ,int ,int);
__global__ void gpu_swap(int* , int* , int , int );

int main(int argc, char* argv[]){
    
    input_file(argv[1]);
   
    block_FW();
    
    write_file(argv[2]);
    
}

void input_file(char * inputfile){
    FILE* fp = fopen(inputfile,"r");
    
    fseek (fp , 0 , SEEK_END);       
    int size = ftell (fp);  
    rewind (fp);
    int num = size/sizeof(int);
    int *pos = (int*) malloc (sizeof(int)*num);
    fread(pos,sizeof(int),num,fp);
    fclose(fp);
    V = pos[0];
    E = pos[1];
    //printf("%d %d\n",V,E);
    //int dist[V][V];
   // dist = (int*)malloc(V*V*sizeof(int));
    cudaHostAlloc((void**)&dist,V*V*sizeof(int),cudaHostAllocMapped);
    #pragma omp parallel 
    {  
        #pragma omp for  collapse(2)
        for(int i=0;i<V;i++)
            for(int j=0;j<V;j++){
                if(i==j) dist[i*V+j] = 0;
                else dist[i*V+j] = INF ;
            }
        #pragma omp for 
            for(int i = 2; i < num; i+=3){
                    dist[pos[i]*V+pos[i+1]] = pos[i+2];
            }
    }
    free(pos); 
}

void write_file(char * writefile){
    FILE *wp  =  fopen(writefile,"wb");
    
   
    fwrite(dist,sizeof(int),V*V,wp);
    
    fclose(wp);
    //free(dist);
    cudaFreeHost(dist);
}
int  ceil(int a, int b){
    return (a+b-1)/b;
}
 void block_FW() {
    int round = ceil(V, BLOCK_SIZE);
   
    // int stream_index=0;
    //  const int num_streams = 4;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++) {
    // 	cudaStreamCreate(&streams[i]);
      // }
    int n_gpu;
    cudaGetDeviceCount(&n_gpu); 
    omp_set_num_threads(n_gpu);
    int up_gpu = (round+n_gpu-1)/n_gpu;
    int down_gpu = round-up_gpu;
    dim3 ThreadPerBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 phase1_blocks(1,1); //phase1  block數
    dim3 phase2_blocks(round,2);
    dim3 phase3_ublocks(round,up_gpu);
    dim3 phase3_dblocks(round,up_gpu);
    cudaError_t  err;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        cudaMalloc(&dev_dist[tid], V*V* sizeof(int));
        err= cudaMemcpy2D(dev_dist[tid],V*sizeof(int), dist, V*sizeof(int),V*sizeof(int),V, cudaMemcpyHostToDevice);
        // if(err!=cudaSuccess){
        //     printf("cuda memcpy error  %s\n",cudaGetErrorString(err));
        // }
        // else{
        //     printf("success");
        // }
        
         //cudaDeviceEnablePeerAccess(1,0);
        // cudaDeviceEnablePeerAccess(0,0);
        for(int r = 0;r < round ; r++){
            //cudaMemcpyAsync(dev_dist+offset, dist+offset, V*V*sizeof(int)/num_streams, cudaMemcpyHostToDevice, stream[i]);
            //#pragma omp barrier
            #pragma omp barrier
            if(r>=up_gpu  &&tid==1){
                int B = (r==round-1 )? (V-r*BLOCK_SIZE):BLOCK_SIZE;
                err = cudaMemcpy2D(dev_dist[0]+r*BLOCK_SIZE*V,V*sizeof(int),dev_dist[1]+r*BLOCK_SIZE*V,V*sizeof(int),V*sizeof(int),B,cudaMemcpyDeviceToDevice);
                if(err!=cudaSuccess){
                    printf("1.cuda memcpy error:%s\n",cudaGetErrorString(err));
                }
			}
			if( r<up_gpu && tid==0){
				err=cudaMemcpy2D(dev_dist[1]+r*BLOCK_SIZE*V,V*sizeof(int),dev_dist[0]+r*BLOCK_SIZE*V,V*sizeof(int),V*sizeof(int),BLOCK_SIZE,cudaMemcpyDeviceToDevice);
                if(err!=cudaSuccess){
                    printf("2.cuda memcpy error:%s\n",cudaGetErrorString(err));
                }
            }
            #pragma omp barrier
            
            phase1<<<phase1_blocks,ThreadPerBlock,BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist[tid],r,V);
            phase2 <<<phase2_blocks,ThreadPerBlock,2*BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist[tid],r,V);
               
           
            #pragma omp barrier
            if (tid == 0 )
                phase3<<<phase3_ublocks,ThreadPerBlock,2*BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist[tid],r,V,up_gpu*tid);
         
         
            if(tid==1){
                phase3<<<phase3_dblocks,ThreadPerBlock,2*BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist[tid],r,V,up_gpu*tid);
            }
            #pragma omp barrier
           
        }
        // if(tid==1)
        //  err=cudaMemcpy2D(dev_dist[0]+up_gpu*BLOCK_SIZE*V, V*sizeof(int), dev_dist[1]+up_gpu*BLOCK_SIZE*V, V*sizeof(int), V*sizeof(int), V-up_gpu*BLOCK_SIZE, cudaMemcpyDeviceToDevice);
        //  if(err!=cudaSuccess){
        //     printf("3.cuda memcpy error:%s\n",cudaGetErrorString(err));
        // }
    }
        if(round!=1){
            cudaMemcpy2D(dist, V*sizeof(int), dev_dist[0], V*sizeof(int),V*sizeof(int), up_gpu*BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(dist+up_gpu*BLOCK_SIZE*V, V*sizeof(int), dev_dist[1]+up_gpu*BLOCK_SIZE*V, V*sizeof(int), V*sizeof(int), V-up_gpu*BLOCK_SIZE, cudaMemcpyDeviceToHost);
        }
        else{
            cudaMemcpy2D(dist, V*sizeof(int), dev_dist[0], V*sizeof(int),V*sizeof(int), V, cudaMemcpyDeviceToHost);
        }
        //cudaMemcpy(dist,dev_dist[0],V*V*sizeof(int),cudaMemcpyDeviceToHost);
        // for (int i=2;i<3;i++){
        //     for(int j=0;j<V;j++)
        //         printf("%d\t",dist[i*V+j]);
        //     printf("\n");
        // }
        cudaFree(dev_dist[0]);
        cudaFree(dev_dist[1]);
       
       
   
    
}

__global__ void phase1(int *Dist,int r,int V){
    extern  __shared__  int shared_block[];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int x = tidy + BLOCK_SIZE*r;
    int y = tidx + BLOCK_SIZE*r;
    //if(x>=V||y>=V) return;
    
    shared_block[tidy*(BLOCK_SIZE)+tidx] = (x<V&&y<V)?Dist[x*V+y]:INF;
    
    __syncthreads();
    #pragma unroll 
    for(int k =0 ;k<BLOCK_SIZE;k++){
        int temp = shared_block[tidy*(BLOCK_SIZE)+k]+ shared_block[k*(BLOCK_SIZE)+tidx];
        shared_block[tidy*(BLOCK_SIZE)+tidx]= min(temp,shared_block[tidy*(BLOCK_SIZE)+tidx]);
     __syncthreads();
    }
     
    if(x<V&&y<V)Dist[x*V+y]= shared_block[tidy*(BLOCK_SIZE)+tidx];
    //__syncthreads();
}
__global__ void phase2(int *Dist ,int r,int V){
    if(blockIdx.x==r)return;
    extern  __shared__  int shared_block[];
    int *pivert = &shared_block[0]; 
    int *share_dist = &shared_block[BLOCK_SIZE*(BLOCK_SIZE)];
    //int *pivert = &shared_block[0]; 
    //int *share_dist = &shared_block[BLOCK_SIZE*BLOCK_SIZE];

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int x = tidy + r*BLOCK_SIZE;
    int y = tidx + r*BLOCK_SIZE;
    
  
        pivert[tidy*BLOCK_SIZE+tidx]=  (x<V && y<V)?Dist[x*V+y]:INF;
    
    
  
    // __syncthreads();
    
    if(blockIdx.y==0)//pivert row (block x不變 y變)  
        y = tidx + blockIdx.x*BLOCK_SIZE;
    
    else//pivert col (block x會變 y不變)          // block: 0 1 2 3 4 5 6 7 ....  blockIdx.y=0 
        x = tidy + blockIdx.x*BLOCK_SIZE;       //        0 1 2 3 4 5 6 7 ....  Idy=1
    if(x>=V || y>=V)return;
   
    
    share_dist[tidy*BLOCK_SIZE+ tidx] =(x<V&&y<V)?Dist[x*V+y]:INF;

    __syncthreads();

    if(blockIdx.y==0){ 
    #pragma unroll 
        for(int k=0;k<BLOCK_SIZE;k++){
            int temp = pivert[tidy*BLOCK_SIZE+k]+share_dist[k*BLOCK_SIZE+tidx];
            share_dist[tidy*BLOCK_SIZE+tidx]=min(temp,share_dist[tidy*BLOCK_SIZE+tidx]);
           
        }
        // __syncthreads();
    }
    else{
    #pragma unroll 32
        for(int k = 0;k<BLOCK_SIZE;k++){
            int temp = share_dist[tidy*BLOCK_SIZE+k]+ pivert[k*BLOCK_SIZE+ tidx];
            //if(share_dist[tidy*BLOCK_SIZE+tidx]>temp)
            share_dist[tidy*BLOCK_SIZE+tidx] = min(temp,share_dist[tidy*BLOCK_SIZE+tidx]);
            
        }
        // __syncthreads() ;
    }
    Dist[x*V+y] = share_dist[tidy*BLOCK_SIZE+tidx];

}



__global__ void phase3(int *Dist ,int r,int V,int offset){
    // /*
    // dim blocksize(32,32)
    // dim gridsize(round,round)
    // */

    int b_idy = blockIdx.y+ offset;
    int b_idx = blockIdx.x;
    if (b_idx== r || b_idy == r) return;
    extern __shared__ int shared_block[];
    int *share_row = &shared_block[0];
    int *share_col = &shared_block[BLOCK_SIZE*(BLOCK_SIZE)];
    //int *share_dist = &shared_block[2*BLOCK_SIZE*(BLOCK_SIZE)];
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
   
    
    const int x = tidy + b_idy*blockDim.y;
    const int y = tidx + b_idx*blockDim.x; 
    const int x_col = tidx + r*(BLOCK_SIZE);
    const int y_row = tidy + r*(BLOCK_SIZE);
    share_row[tidy*(BLOCK_SIZE) +tidx] =(x<V && x_col<V) ? Dist[x*V + x_col] :INF;
    share_col[tidy*(BLOCK_SIZE) +tidx] =(y_row<V && y<V) ? Dist[y_row*V + y] : INF;
    //share_dist[tidy*(BLOCK_SIZE) +tidx] = (x<V && y<V) ? Dist[x*V + y] : INF;
     __syncthreads();
     
     
    if(x>=V||y>=V)return;
        //int d_temp = share_dist[tidy*(BLOCK_SIZE) +tidx];
        int d_temp = Dist[x*V+y];
        //int index =tidy*(BLOCK_SIZE) +tidx;
        #pragma unroll 
        for(int k=0;k<BLOCK_SIZE;k++){
            int temp = share_row[tidy*(BLOCK_SIZE)+k] + share_col[k*(BLOCK_SIZE)+tidx];
            //d_temp=temp;
            // if(temp<d_temp)
            //     d_temp= temp;
            d_temp = min(d_temp,temp);
            //share_dist[index] = min(share_dist[index],temp);
        }
        //Dist[x*V+y] = share_dist[index];
        Dist[x*V+y] = d_temp;
    
}
