#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include "cl.hpp"
#include "util.hpp"
#include <vector>

using namespace std;
using namespace cl;

#define N 1024

vector<float> h_matrixmul(vector<float> A, vector<float> B, vector<float> C) {
	int i,j, k;

	vector<float> ACopy(A);
	vector<float> BCopy(B);
	vector<float> CCopy(C);

	for(i=0; i<N; ++i)
	{
		for(j=0; j<N; ++j)
		{
			float temp = 0.0;
			for(k=0; k<N; ++k)
			{
				temp += BCopy[i * N + k] * CCopy[k * N + j];
			}

			ACopy[i * N + j] = temp;
		}
	}

	return ACopy;
}

vector<float> d_matrixmul(vector<float> A, vector<float> B, vector<float> C) {
	Context context(CL_DEVICE_TYPE_DEFAULT);

	vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	Program program(context, util::loadProgram("kernels.cl"));
	CommandQueue queue(context);

	vector<float> Acpy(A);

	try {
		program.build();
		auto mmul = cl::make_kernel<int, Buffer, Buffer, Buffer, LocalSpaceArg>(program, "matrixmul");

		// Use the Draconion naming convention of d_* for declaring device memory
		Buffer d_A, d_B, d_C;
		LocalSpaceArg CWork = Local(sizeof(float) * N);

		d_A = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);
		d_B = Buffer(context, begin(B), end(B), true);
		d_C = Buffer(context, begin(C), end(C), true);

		printf("Launching Kernel!\n");

		// Launch the specified kernel with the specified dimensions and arguments
		mmul(EnqueueArgs(queue, NDRange(N), NDRange(64)), N, d_A, d_B, d_C, CWork);

		queue.finish();

		// Once done we can copy the data from the operation
		copy(queue, d_A, begin(Acpy), end(Acpy));
	} 
	catch (cl::Error err) 
	{
		if(err.err() == CL_BUILD_PROGRAM_FAILURE) {
			string build_error = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << "Error compiling Kernel code: " << build_error << endl;
		}
		else {
	    	std::cout << "OpenCL Error: " << err.what() << " returned " << err.err() << std::endl;
	    	std::cout << "Check cl.h for error codes." << std::endl;
	    	exit(-1);
		}
	}

	return Acpy;
}

// We Will be calculating the equation A = B * C

int main(int argc, char *argv[]) {
	util::Timer timer;
	double rtime;	

	printf("Starting...\n");

	// We will be calculating calculations of square NxN matrices
	vector<float> A(N * N);
	vector<float> B(N * N);
	vector<float> C(N * N);

	// Initialise Matrices with random data
	for(int i=0; i < N * N; ++i)
	{
		B[i] = rand() % 100;
		C[i] = rand() % 100;
	}

	vector<float> h_result;
	vector<float> d_result;

	// HOST CPU MATRIX MULTIPLY
	timer.reset();
	printf("Performing matrix multiply on the host...\n");
	h_result = h_matrixmul(A, B, C);
	rtime = static_cast<double>(timer.getTimeMilliseconds() / 1000.0);		
	printf("Host computation complete in %f seconds\n", rtime);

	// DEVICE GPU MATRIX MULTIPLY
	timer.reset();
	printf("Performing matrix multiply on the device gpu...\n");
	d_result = d_matrixmul(A, B, C);
	rtime = static_cast<double>(timer.getTimeMilliseconds() / 1000.0);
	printf("The Kernels finished in %f seconds\n", rtime);
	
	bool result = true;
	for(uint i=0; i<d_result.size(); ++i) {
		result &= h_result[i] == d_result[i];
	}

	if(result)
		printf("Success!\n");
	else
		printf("Validation error\n");
}
