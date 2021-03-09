#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <assert.h>
#include <string.h>

#define DISPATCH 1
#define  FINISH  2
#define COMPLETE 3
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void master(int,int*,int*,int);
void slave(int,int*);
void mandelbrot(int,int,int*);
int cal_(double,double ,int,int);
void write_image_buffer(int*, int*,int);

int row_complete,nrank;

int master_n;
int iters,height,width;
double left,right,lower,upper; 
double delta_y ;
double delta_x ;



int main (int argc, char* argv[]){

    int rank,ntask;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
    MPI_Comm_size (MPI_COMM_WORLD, &ntask); 
    MPI_Status status;
    master_n = ntask-1;

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    delta_y =((upper - lower) / height);
    delta_x =((right-left)/width);

    int num_row = omp_get_max_threads() ;
    /* allocate memory for image */
    int *image = (int*)malloc(width * height * sizeof(int));
    int *local_image =  (int*)malloc((num_row*width+1)*sizeof(int));
    assert(image);
    if (rank==master_n){
        master(num_row,image,local_image,ntask);
        write_png(filename, iters, width, height, image);
    }

    else {
        slave(num_row,local_image);
    }


    
    MPI_Finalize();
    free(image);
    free(local_image);
    return 0;
}


void master(int num_row,int *image,int*image_buffer,int ntask){
    MPI_Status status;
    MPI_Request request; 
    int job=0;
    row_complete = 0;
    

    for( ;job<master_n&&row_complete<height;job++){
        MPI_Isend(&row_complete,1,MPI_INT,job,DISPATCH,MPI_COMM_WORLD,&request);
        row_complete+=num_row;
    }
    do{
        
        MPI_Recv(image_buffer,width*num_row+1,MPI_INT,MPI_ANY_SOURCE,FINISH,MPI_COMM_WORLD,&status);
        int slave = status.MPI_SOURCE;
        write_image_buffer(image,image_buffer,num_row);
        job--;
        if(row_complete<height){
            MPI_Isend(&row_complete,1,MPI_INT,slave,DISPATCH,MPI_COMM_WORLD,&request);
            row_complete+=num_row;
            job++;
        }
        else{
            MPI_Isend(&row_complete,1,MPI_INT,slave,COMPLETE,MPI_COMM_WORLD,&request);
        }
    }while(job>0);
            
}

void slave(int num_row,int *local_image){
    MPI_Status status;
    MPI_Request request;
    int col;
    MPI_Recv(&col,1,MPI_INT,master_n,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
    while(status.MPI_TAG==DISPATCH){
         //if(height - col< num_row) num_row = height-col;
         mandelbrot(col,num_row,local_image);
         //printf("%d \n",col);
         
         //printf("  col:%d\n",local_image[0]);
         MPI_Isend(local_image,num_row*width+1,MPI_INT,master_n,FINISH,MPI_COMM_WORLD,&request);
         MPI_Recv(&col,1,MPI_INT,master_n,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
    }
    
    

}


void mandelbrot(int col,int num_row,int *local_image){
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int k=0;k<num_row;k++){
        for(int i=0; i<width;i++ ){
            if(col+k<height){
            double y0 = (col+k) * delta_y + lower;
            double x0 = i * delta_x + left;
            local_image[1+k*width+i] =cal_(x0,y0,i,k);
            }
            
            //printf("thread:%d (%d,%d) repeats:%d \n",thread_num,k,i,cal_(x0,y0,i,k));

        }
         
    } 

    local_image[0]=col; 
}

int cal_(double x0,double y0,int i,int k){
    //每個thread算一列
       
        
       
        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;//temp:Z_real_next,  x0=C_real
            y = 2 * x * y + y0;//Z_imag_next 
            x = temp; 
            length_squared =  x * x +y * y;
            ++repeats;
        }
        //printf("repeats:%d\n");
        return repeats;
        
}

void write_image_buffer(int *image_, int *buffer,int num_row ){
   int i,j,col=0;
   col = buffer[0];
   for (i=0;i<num_row;i++){
        for(j=0;j<width;j++){
            if (col<height){
                
                image_[col*width+j] = buffer[1+i*width+j];
            }
        }
        col++;
    }
    
}