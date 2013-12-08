
// Performs a matrix multiplication on a NxN matrix
__kernel void matrixmul(const int N, __global float *A, __global float *B, __global float *C, __local float *CWork) 
{
	int i = get_global_id(0);
	int j, k;

	int iloc = get_local_id(0);
	int nloc = get_local_size(0);

	__private float BWork[1024];

	for(k=0; k<N; ++k) {
		BWork[k] = B[i * N + k];
	}

	for(j=0; j<N; ++j) {

		for(k=iloc; k<N; k+=nloc)
			CWork[k] = C[k * N + j];

		barrier(CLK_LOCAL_MEM_FENCE);

		float temp = 0.0f;

		for(k=0; k<N; ++k) {
			temp += BWork[k] * CWork[k];
		}

		A[i * N + j] = temp;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
