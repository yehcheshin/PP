
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <emmintrin.h>

#define N 2

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

//global variable
long long num_threads, iters;
double left, right, lower, upper;
long long  height, width;
int* image;
long long offset;
double delta_y;
double delta_x;

pthread_mutex_t  mutex_offset;

/* normal */
void threading(double x0, double y0, int i, int count) {
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
    image[count * width + i] = repeats;
}

void * mandelbrot_set(void *threadid){
    int *tid = (int*)threadid;
    int count;

    do{
        pthread_mutex_lock(&mutex_offset);
        offset++;
        count = offset-1;
        pthread_mutex_unlock(&mutex_offset);

        if(offset > height) break;

        /* vectorization */
        double y0 = count * delta_y + lower;
        int i;

        __m128d _x0, _y0, _x, _y, _length_sq, _tmp, _x2, _y2, _four, _mask1;
        __m128d _n, _c, _one, _iters, _mask2;

        _y0 = _mm_set_pd(y0,y0);
        _four = _mm_set_pd(4.0,4.0);
        _one = _mm_set_pd(1.0,1.0);
        _iters = _mm_set_pd(iters,iters);

        for (i = 0; i + N < width; i += N) {
            _n = _mm_setzero_pd();
            _x0 = _mm_set_pd((i+1) * delta_x + left,i * delta_x + left);
            _x = _mm_setzero_pd();
            _y = _mm_setzero_pd();
            _length_sq = _mm_setzero_pd();
            _mask1 = _mm_cmplt_pd(_length_sq, _four);
            _mask2 = _mm_and_pd(_mm_cmplt_pd(_n,_iters),_mask1);
            //  repeat:
                // double temp = x * x - y * y + x0;
            while(_mm_movemask_pd(_mask2)){
                _x2 = _mm_mul_pd(_x, _x);
                _y2 = _mm_mul_pd(_y, _y);
                _tmp = _mm_sub_pd(_x2, _y2);
                _tmp = _mm_add_pd(_tmp, _x0);

                // x = tmp
                _x  = _tmp;
                _x2 = _mm_mul_pd(_x, _x);

                // length_squared =  x * x + y * y;
                _length_sq = _mm_add_pd(_x2, _y2);

                // if condition need cinsider iters and length -> need mask to do it
                

                // AND with _one -> get the correct iter
                _c = _mm_and_pd(_one, _mask2);

                _n = _mm_add_pd(_n, _c);

                // length_squared < 4
                _mask1 = _mm_cmplt_pd(_length_sq, _four);

                // repeats < iters
                _mask2 = _mm_cmplt_pd(_n,_iters);

                // if both condition fit then do
                _mask2 = _mm_and_pd(_mask2, _mask1);


            }
            // reduce the condition
            // if (_mm_movemask_pd(_mask2) > 0)
            //     goto repeat;

            // assgin to image
            double temp[2];
            _mm_store_pd(&temp[i],_n);
            image[count * width + i] = temp[i];
        }

        /* normal threading */
        for (i = i-N; i < width; i++) {
            double x0 = i * delta_x + left;
            threading(x0, y0, i, count);
        }

    } while(count < height);
pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);
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

    /* initialzation of scaling stuff*/
    delta_y =((upper - lower) / height);
    delta_x =((right-left)/width);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    void *status;
    pthread_t threads[num_threads];
    pthread_mutex_init(&mutex_offset,NULL);

    int rc;
    int ID[num_threads];

    for(int t=0;t<num_threads;t++){
        ID[t] = t;
        rc = pthread_create(&threads[t], NULL,mandelbrot_set, (void*)&ID[t]);
    }
    for (int t=0;t<  num_threads ;t++){
		pthread_join(threads[t],&status);
	}
    /* draw and cleanup */

    write_png(filename, iters, width, height, image);
    free(image);

    return 0;
}
