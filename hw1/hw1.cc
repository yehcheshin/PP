#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include  <algorithm>

#define SWAP(x,y){float temp;temp=x; x=y ; y=temp;}


float* Odd_Even_Sort(float [],int,int,int,int);
bool compare_and_swap(float[],int,int,float[],int,int,bool);
int *even_neighbor_buffer ;
int *odd_neighbor_buffer;
int *array_recv;
float *lr_buff;

int cmpfunc (const void * a, const void * b)
{
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}


int main (int argc, char* argv[]){

//inputData
int N = atoi(argv[1]);
int i, rank, size, num_elem,buffer_size;
int count;
float array[N],*final_array;
float data[N];
MPI_Init(&argc, &argv); 
MPI_File f;
MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
MPI_Comm_size (MPI_COMM_WORLD, &size); 
MPI_Status status;



//processe與input數量判斷
if(rank>=N)MPI_PROC_NULL;
//每個process 分配個數（餘數平攤給前面r個process)
num_elem = N/size;
int r = N%size;
if(rank < r )num_elem= num_elem +1;

float *local_arr=(float*)malloc(num_elem*sizeof(float));
//檔案讀寫offset設定
int rank_num ;
if(r==0)rank_num = rank*num_elem;
else{
    if(rank>r)rank_num = rank*num_elem+r;
    else rank_num = rank*num_elem;
    if (rank==r&&rank!=0)rank_num = rank*(num_elem+1);
}
//檔案讀取

MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

MPI_File_read_at(f, sizeof(float)*rank_num,local_arr,num_elem,MPI_FLOAT,MPI_STATUS_IGNORE);
MPI_File_close(&f);

final_array = Odd_Even_Sort(local_arr,num_elem,rank,size,r);

MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY| MPI_MODE_CREATE , MPI_INFO_NULL, &f);
MPI_File_write_at(f, sizeof(float)*rank_num,final_array,num_elem,MPI_FLOAT,&status);
MPI_File_close(&f);
free(final_array);

MPI_Finalize();
return 0;
}

float*  Odd_Even_Sort(float local_arr[],int arr_size,int rank,int size,int r){
    bool sorted= false,temp;
    int left_neighbor,right_neighbor;
    int right_size,left_size;
    
   //暫存right_neighbor data 之buffer
    float *temp_buffer = (float*) malloc(arr_size*sizeof(float));
    MPI_Status status;
    MPI_Request requests;
   //設定process左右鄰居  
    right_neighbor = rank+1;
    left_neighbor = rank-1;
    if(right_neighbor==size)right_neighbor=MPI_PROC_NULL;
    if(left_neighbor==-1) left_neighbor=MPI_PROC_NULL;
    
    if (right_neighbor >=r && rank <r) lr_buff = (float*)malloc((arr_size+arr_size-1)*sizeof(float));
    else lr_buff=(float*)malloc(2*arr_size*sizeof(float));
    
    //每個process 先對local_arr做排序
    
    std::sort(local_arr,local_arr+arr_size);
    
    while(!sorted){
      temp=sorted = true;
      //even phase
        if (rank%2!=0){//rank為奇數之process,將datat傳給左邊process
          MPI_Isend(local_arr,arr_size,MPI_FLOAT,left_neighbor,0,MPI_COMM_WORLD,&requests);
         
          MPI_Recv(local_arr,arr_size,MPI_FLOAT,left_neighbor,2,MPI_COMM_WORLD,&status);
        }
        else{ //rank 是偶數
              if(right_neighbor!=MPI_PROC_NULL){
                int even_count;
                
                MPI_Recv(temp_buffer,arr_size,MPI_FLOAT,right_neighbor,0,MPI_COMM_WORLD,&status);
                MPI_Get_count(&status,MPI_INT,&even_count);
                 //做完compare_and_swap,左邊(偶數rank)process將data回傳給右邊process
                sorted = compare_and_swap(local_arr,arr_size,rank,temp_buffer,even_count,right_neighbor,sorted);
                temp = sorted;
              }
            }
        
        MPI_Barrier(MPI_COMM_WORLD);
      
      
     
    
      //odd phase
          if(rank%2==0 && left_neighbor!=MPI_PROC_NULL){//rank為偶數之process,將data傳給左邊process
              MPI_Isend(local_arr,arr_size,MPI_FLOAT,left_neighbor,1,MPI_COMM_WORLD,&requests);
              MPI_Recv(local_arr,arr_size,MPI_FLOAT,left_neighbor,2,MPI_COMM_WORLD,&status);
            }
          else{//奇數rank之process接收右邊process之data
                  if(right_neighbor!=MPI_PROC_NULL && left_neighbor!=MPI_PROC_NULL) {
                     int odd_count;
                    //
                     MPI_Recv(temp_buffer,arr_size,MPI_FLOAT,right_neighbor,1,MPI_COMM_WORLD,&status);
                    
                     MPI_Get_count(&status,MPI_INT,&odd_count);
                    //執行compare_and_swap,並回傳給右邊的process
                    sorted = compare_and_swap(local_arr,arr_size,rank,temp_buffer,odd_count,right_neighbor,sorted);
                    temp = sorted;
                  }
            }
            
        MPI_Barrier(MPI_COMM_WORLD);
      //若所有陣列都排序過且無做交換，跳出迴圈
       MPI_Allreduce(&temp,&sorted,1,MPI_CHAR,MPI_BAND,MPI_COMM_WORLD);
    }  
    free(temp_buffer);
    return local_arr;
}

bool compare_and_swap(float arr[],int arr_size,int rank,float right_arr[],int right_size,int right_neighbor,bool sorted ){
 MPI_Status status;
 MPI_Request requests;
  int i=0,j=0,k=0,h= arr_size+right_size;
  
  float r_min = right_arr[0];
  float  l_max = arr[arr_size -1];
  //如果左邊最大元素比右邊小＝>直接回傳 sorted = true否則回傳false
  if (r_min>=l_max){
    sorted = true;
  }
  
  else {
      //採用mergesort方式來做兩個已排序過的陣列比較 做排列
      while(i<arr_size && j<right_size){
        if(arr[i]<=right_arr[j]){
         lr_buff[k++] = arr[i++];
          
         }
         else{
           sorted = false;
           lr_buff[k++] = right_arr[j++];
         }
      }
     
      while(i<arr_size)lr_buff[k++] = arr[i++];
      while(j<right_size)lr_buff[k++]  = right_arr[j++];
      for(int i=0;i<arr_size;i++)arr[i] = lr_buff[i];
      for(int i=0;i<right_size;i++)right_arr[i] = lr_buff[i+arr_size];
  }
  MPI_Isend(right_arr,right_size,MPI_FLOAT,right_neighbor,2,MPI_COMM_WORLD,&requests);
  
  return sorted;
  
}






