#include    <stdio.h>
#include    <stdlib.h>
#include    <omp.h>

int V;
int E;

int main(int argc, char* argv[]){
   
    FILE* fp = fopen(argv[1],"r");
    if (!fp){
        perror("無法讀取檔案");
        return  EXIT_FAILURE;
    }
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
    int **dist;
    dist = (int**)malloc(V*sizeof(int*));
    for (int i=0;i<V;i++)
        dist[i] = (int*)malloc(V*sizeof(int));
   
    #pragma omp parallel 
    {  
        #pragma omp for schedule(static,10) collapse(2)
        for(int i=0;i<V;i++)
            for(int j=0;j<V;j++){
                if(i==j) dist[i][j] = 0;
                else dist[i][j] = 1073741823 ;
            }
        #pragma omp for schedule (static,10)
            for(int i = 2; i < num; i+=3){
                    dist[pos[i]][pos[i+1]] = pos[i+2];
            }
    }
    //printf("%d ",thread_num);

    int temp,val;
    #pragma omp parallel private(val,temp) shared(dist) 
    {   
        for (int k=0;k<V;k++)
        #pragma  omp for
             for(int i = 0;i< V;i++){
                 temp = dist[i][k];
                for(int j = 0;j<V;j++){
                      val = temp+dist[k][j];
                    if (dist[i][j] >val)
                        dist[i][j] = val;
                }
                
             }
             
                
                    // if (dist[i][j]> dist[i][k]+dist[k][j])
                    //     dist[i][j] = dist[i][k]+dist[k][j];
    }
 

    
    FILE *wp  =  fopen(argv[2],"wb");
    if (!wp){
        perror("無法寫入檔案");
        return EXIT_FAILURE;
    }
    #pragma omp parallel for ordered
    for(int i=0;i<V;i++){
        #pragma omp ordered
        fwrite(dist[i],sizeof(int),V,wp);
    }
    
    fclose(wp);


    free(pos);  
    for(int i=0;i<V;i++)
        free(dist[i]);
    
}