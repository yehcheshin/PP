#include    <stdio.h>
#include    <stdlib.h>
#include    <cuda_runtime.h>
#define BLOCK_SIZE 32
#define INF 1073741823
int V;
int E;
int *dist;
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
__global__ void phase1(int *Dist,int r,int V);
__global__ void phase2(int *Dist ,int r,int V);
__global__ void phase3(int *Dist ,int r,int V);

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
    dist = (int*)malloc(V*V*sizeof(int));
    //cudaMallocHost((void**)&dist,V*V*sizeof(int));
    for(int i=0;i<V;i++){
        for(int j=0;j<V;j++){
            if(i==j) dist[i*V+j] = 0;
            else dist[i*V+j] = INF ;
        }
    }
    for(int i = 2; i < num; i+=3){
            dist[pos[i]*V+pos[i+1]] = pos[i+2];
    }
    free(pos); 
}

void write_file(char * writefile){
    FILE *wp  =  fopen(writefile,"wb");
    
   
    fwrite(dist,sizeof(int),V*V,wp);
    
    fclose(wp);
    free(dist);
}
int  ceil(int a, int b){
    return (a+b-1)/b;
}
 void block_FW() {
    int round = ceil(V, BLOCK_SIZE);
    int *dev_dist;
    // int stream_index=0;
    //  const int num_streams = 4;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++) {
    // 	cudaStreamCreate(&streams[i]);
  	// }
  
    cudaMalloc(&dev_dist, V*V* sizeof(int));
  
    //memcpy(host1, dist, V*V*sizeof(int));

    dim3 ThreadPerBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 phase1_blocks(1,1); //phase1  block數
    dim3 phase2_blocks(round,2);
    dim3 phase3_blocks(round,round);
    
    //cudaMemcpyAsync(dev_dist,dist,V*V*sizeof(int), cudaMemcpyHostToDevice);
    // for(int r = 0;r < round ; r++){
    
    //     phase1<<<phase1_blocks,ThreadPerBlock,block_size*block_size*sizeof(int)>>>(dev_dist,r,V,block_size);
    //     phase2 <<<phase2_blocks,ThreadPerBlock,2*block_size*block_size*sizeof(int)>>>(dev_dist,r,V,block_size);
    //     phase3<<<phase3_blocks,ThreadPerBlock,2*block_size*block_size*sizeof(int)>>>(dev_dist,r,V,block_size);
        
    //     // cudaStreamSynchronize(current_stream);
    //     // stream_index++;
    // }
    // cudaEvent_t start,end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);

    
    cudaMemcpy(dev_dist,dist,V*V*sizeof(int), cudaMemcpyHostToDevice);
    //cudaEventRecord(start);
    for(int r = 0;r < round ; r++){
        //cudaMemcpyAsync(dev_dist+offset, dist+offset, V*V*sizeof(int)/num_streams, cudaMemcpyHostToDevice, stream[i]);
        phase1<<<phase1_blocks,ThreadPerBlock,BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist,r,V);
        phase2 <<<phase2_blocks,ThreadPerBlock,2*BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist,r,V);
        phase3<<<phase3_blocks,ThreadPerBlock,2*BLOCK_SIZE*(BLOCK_SIZE)*sizeof(int)>>>(dev_dist,r,V);
         // cudaStreamSynchronize(current_stream);
        // stream_index++;
       
    }
    //cudaEventRecord(end);
    cudaMemcpy(dist,dev_dist,V*V*sizeof(int),cudaMemcpyDeviceToHost);
    //cudaEventSynchronize(end);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, end);
    cudaFree(dev_dist);
    //printf("time:%lf\n",milliseconds);
    // for (int i = 0; i < num_streams; i++) {
    // 	cudaStreamDestroy(streams[i]);
    //   }
    //cudaMemcpyAsync(dist,dev_dist,V*V*sizeof(int),cudaMemcpyDeviceToHost,0);
   
    
}


__global__ void phase1(int *Dist,int r,int V){
    extern  __shared__  int shared_block[];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int x = tidy + BLOCK_SIZE*r;
    int y = tidx + BLOCK_SIZE*r;
    //if(x>=V||y>=V) return;
    if( check_bound(x,y,V))
        shared_block[tidy*(BLOCK_SIZE)+tidx] = Dist[x*V+y];
    else
        shared_block[tidy*(BLOCK_SIZE)+tidx] = INF;
    __syncthreads();
    #pragma unroll 32
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
    
    if (x<V && y<V)
        pivert[tidy*BLOCK_SIZE+tidx]=Dist[x*V+y];
    
    else 
        pivert[tidy*BLOCK_SIZE+tidx]=INF;
  
    // __syncthreads();
    
    if(blockIdx.y==0)//pivert row (block x不變 y變)  
        y = tidx + blockIdx.x*BLOCK_SIZE;
    
    else//pivert col (block x會變 y不變)          // block: 0 1 2 3 4 5 6 7 ....  blockIdx.y=0 
        x = tidy + blockIdx.x*BLOCK_SIZE;       //        0 1 2 3 4 5 6 7 ....  Idy=1
    if(x>=V || y>=V)return;
   
    if(x<V&&y<V)
        share_dist[tidy*BLOCK_SIZE+ tidx] =Dist[x*V+y];
    else
        share_dist[tidy*BLOCK_SIZE+ tidx]=INF;
    __syncthreads();

    if(blockIdx.y==0){ 
    #pragma unroll 32
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



__global__ void phase3(int *Dist ,int r,int V){
    /*
    dim blocksize(32,32)
    dim gridsize(round,round)
    */
    if (blockIdx.x == r || blockIdx.y == r) return;
    extern __shared__ int shared_block[];
    int *share_row = &shared_block[0];
    int *share_col = &shared_block[BLOCK_SIZE*(BLOCK_SIZE)];
    //int *share_dist = &shared_block[2*BLOCK_SIZE*(BLOCK_SIZE)];
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
   
    
    const int x = tidy + blockIdx.y*blockDim.y;
    const int y = tidx + blockIdx.x*blockDim.x; 
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
        #pragma unroll 32
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
