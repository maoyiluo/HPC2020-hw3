#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <omp.h>

const int thread_num = 4;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan

  if (n == 0) return;
  prefix_sum[0] = 0;

  int chunk_size = n/thread_num;

  // initialized the correction array
  long* correction = (long*) malloc(thread_num * sizeof(long));
  for(int i = 0; i < thread_num; i++){
    correction[i] = 0;
  }

  // Scanning
  #pragma omp parallel num_threads(thread_num)
  {
    int thread_id = omp_get_thread_num();
    
    // chunk_size + 1 make sure we don't have to wrap the thread.
    #pragma omp for schedule(static,chunk_size+1)
    for (long i = 1; i < n; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      correction[thread_id]+= A[i-1];
    }

    // wait for other thread
    #pragma omp barrier

    // correct the prefix_sum
    #pragma omp for schedule(static,chunk_size+1)
    for(long i = 1; i < n; i++){
      if(thread_id!=0)
        for(int j = 0; j < thread_id; j++){
          prefix_sum[i] += correction[j];
        }
    }
  } 
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) {B0[i] = 0; B1[1] = 0;}

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
