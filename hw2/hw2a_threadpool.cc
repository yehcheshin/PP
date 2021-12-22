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
#include <unistd.h>

#define ThreadPoolSize 256

// sturct for arguments
typedef struct threadArgs
{
    double x0;
    double y0;
    int i;
    int j;
} threadArgs;

// init
pthread_mutex_t mutexQueue;
pthread_cond_t condQueue;
double left, right, lower, upper;
int height, width, iters;
bool isFull = false;
int *image;
int NHTHREADS;
volatile int taskCount = 0;
int count = 0;
threadArgs tArgs[ThreadPoolSize];

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
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
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
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

// exe function
void mandelbrotSet(threadArgs data)
{
    int repeats = 0;
    double x0 = data.x0;
    double y0 = data.y0;
    double x = 0;
    double y = 0;
    int i = data.i;
    int j = data.j;
    double length_squared = 0;

    while (repeats < iters && length_squared < 4)
    {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }
    // printf(" (%d,%d),count:%d\n",i,j,repeats);
    image[j * width + i] = repeats;
}

// start exe
void *mandelbrot(void *arg)
{
    while (count < width * height)
    {
        threadArgs data;
        bool executable = false;
        pthread_mutex_lock(&mutexQueue);

        while (taskCount == 0 && !isFull)
        {
            pthread_cond_wait(&condQueue, &mutexQueue);
        }

        if (taskCount > 0)
        {
            data = tArgs[0];
            for (int i = 0; i < taskCount - 1; i++)
            {
                tArgs[i] = tArgs[i + 1];
            }
            taskCount--;
            executable = true;
        }

        count++;

        if (count == width * height)
        {
            taskCount--;
            isFull = true;
            pthread_cond_broadcast(&condQueue);
        }
        if (isFull && !executable)
        {
            pthread_mutex_unlock(&mutexQueue);
            pthread_exit(NULL);
        }
        pthread_mutex_unlock(&mutexQueue);
        mandelbrotSet(data);
    }
}

// store the arg
void submitPool(threadArgs argument)
{
    pthread_mutex_lock(&mutexQueue);
    tArgs[taskCount] = argument;
    taskCount++;
    pthread_mutex_unlock(&mutexQueue);
    pthread_cond_signal(&condQueue);
}

int main(int argc, char **argv)
{
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NHTHREADS = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int *)malloc(width * height * sizeof(int));
    assert(image);
    // printf("%s",filename);
    // init
    pthread_t threadID[NHTHREADS];
    pthread_mutex_init(&mutexQueue, NULL);
    pthread_cond_init(&condQueue, NULL);

    for (int i = 0; i < NHTHREADS; i++)
    {
        pthread_create(&threadID[i], NULL, &mandelbrot, NULL);
    }

    /* mandelbrot set */
    for (int j = 0; j < height; ++j)
    {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i)
        {
            double x0 = i * ((right - left) / width) + left;

            threadArgs argument = {
                .x0 = x0,
                .y0 = y0,
                .i = i,
                .j = j};

            while (taskCount == ThreadPoolSize)
            {
            }
            submitPool(argument);
        }
    }

    // wait to the end
    for (int i = 0; i < NHTHREADS; i++)
    {
        // printf("before done thread %d\n", i);
        pthread_join(threadID[i], NULL);
        // printf("after done thread %d\n", i);
    }

    // destroy
    pthread_mutex_destroy(&mutexQueue);
    pthread_cond_destroy(&condQueue);

    // printf("we are almost done");

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
