CC=g++
CFLAGS=-Wall -std=c++11

INCLUDES=-I/usr/local/cuda-5.5/include
LDFLAGS=-L/usr/lib/nvidia-331

EXECUTABLE=matrix_mul.exe

all: $(EXECUTABLE)

$(EXECUTABLE): matrix_mul.cpp
		$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) matrix_mul.cpp -o $@ -lOpenCL
