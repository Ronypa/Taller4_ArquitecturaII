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

// Serial operations of multiplying two arrays and a constant
void operations_ser(float32_t * a, float32_t * b, float32_t * c, int size){
    double start_time, run_time;
    float div = 0.001f;
    start_time = omp_get_wtime();
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i]*b[i]*a[i]*div*b[i]*div*a[i]*div*b[i]*div*div*a[i]*b[i]*div*div*div*div;
    }
    run_time = omp_get_wtime() - start_time;
    printf("Operations in serial with size of %d were completed in %f seconds \n", size, run_time);
}

// Parallel operations of multiplying two arrays and a constant with OpenMP and NEON Intrinsics
void operations_par(float32_t * a, float32_t * b, float32_t * c, int size){
    double start_time, run_time;
    float32x4_t aTmp; 
    float32x4_t bTmp;
    float32x4_t cTmp;
    float32x4_t div = vdupq_n_f32(0.001f);
    omp_set_num_threads(5);
    #pragma omp parallel
    {
        start_time = omp_get_wtime();
        #pragma omp for private(aTmp, bTmp, cTmp)
        for (int i = 1; i <= size; i += 4)
        {
			aTmp = vld1q_f32(a + i);
            bTmp = vld1q_f32(b + i);
            
            cTmp = vmulq_f32(aTmp, bTmp);
            cTmp = vmulq_f32(cTmp, aTmp);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, bTmp);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, aTmp);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, bTmp);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, aTmp);
            cTmp = vmulq_f32(cTmp, bTmp);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, div);
            cTmp = vmulq_f32(cTmp, div);

            vst1q_f32(c + i, cTmp);
        }
    }
    run_time = omp_get_wtime() - start_time;
    printf("Operations in parallel with size of %d were completed in %f seconds \n", size, run_time);
}


int main(){
    srand(time(NULL));
    float32_t a1[size1], b1[size1], c1[size1];

    for (int i = 0; i < size1; i++)
    {
        a1[i] = (float) rand()/(RAND_MAX/9999);
        b1[i] = (float) rand()/(RAND_MAX/9999);
    }
    printf("\n");
    operations_ser(a1, b1, c1, size1);
    operations_par(a1, b1, c1, size1);
    printf("\n");
    operations_ser(a1, b1, c1, size2);
    operations_par(a1, b1, c1, size2);
    printf("\n");
    operations_ser(a1, b1, c1, size3);
    operations_par(a1, b1, c1, size3);
    printf("\n");
}