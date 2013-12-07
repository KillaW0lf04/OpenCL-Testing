
// Performs a matrix multiplication on a NxN matrix
__kernel void matrixmul(const int N, __global float *A, __global float *B, __global float *C) 
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k;
	float temp = 0.0f;

	for(k=0; k<N; ++k) {
		temp += B[i * N + k] * C[k * N + j];
	}

	A[i * N + j] = temp;
}