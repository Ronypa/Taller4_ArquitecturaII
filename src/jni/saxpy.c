/// Tecnologico de Costa Rica
/// Area Academica de Ingenieria en Computadores
/// Arquitectura de Computadores II
/// Taller 4
/// Rony Paniagua Chacon 2015013562
/// Semestre II 2019

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

static int size1 = 600000;
static int size2 = 360000;
static int size3 = 240000;

// Serial SAXPY operation
void saxpy_simple(float32_t x, float32_t * a, float32_t * b, float32_t * c, int size){
    double start_time, run_time;
    start_time = omp_get_wtime();
    for (int i = 0; i < size; i++)
    {
        c[i] = (x * a[i]) + b[i];
    }
    run_time = omp_get_wtime() - start_time;
    printf("Saxpy serial and size of %d was completed in %f seconds \n", size, run_time);
}

// Parallel SAXPY operations using OpenMP and NEON Instrinsics
void saxpy_omp(float32x4_t x, float32_t * a, float32_t * b, float32_t * c, int size){
    double start_time, run_time;
    omp_set_num_threads(5);
    float32x4_t aTmp; 
    float32x4_t bTmp;
    float32x4_t cTmp;
    #pragma omp parallel
    {
        start_time = omp_get_wtime();
        #pragma omp for private(aTmp, bTmp, cTmp)
        for (int i = 0; i < size; i+=4)
        {

			aTmp = vld1q_f32(a + i);
            bTmp = vld1q_f32(b + i);
            
            cTmp = vmulq_f32(x, aTmp);
            cTmp = vaddq_f32(cTmp, bTmp);

			vst1q_f32(c + i, cTmp);
        }
    }
    run_time = omp_get_wtime() - start_time;
    printf("Saxpy parallel and size of %d was completed in %f seconds \n", size, run_time);
    }

int main(){
    srand(time(NULL));

    float x_ser = (float) rand()/(RAND_MAX/9999);
    float32x4_t x_par = vdupq_n_f32(x_ser);

    float32_t a1[size1], b1[size1], c1[size1];

    for (int i = 0; i < size1; i++)
    {
        a1[i] = (float) rand()/(RAND_MAX/9999);
        b1[i] = (float) rand()/(RAND_MAX/9999);
    }
    printf("\n");
    saxpy_simple(x_ser, a1, b1, c1, size1);
    saxpy_omp(x_par, a1, b1, c1, size1);    
    printf("\n");
    saxpy_simple(x_ser, a1, b1, c1, size2);
    saxpy_omp(x_par, a1, b1, c1, size2);
    printf("\n");
    saxpy_simple(x_ser, a1, b1, c1, size3);
    saxpy_omp(x_par, a1, b1, c1, size3);
    printf("\n");
}