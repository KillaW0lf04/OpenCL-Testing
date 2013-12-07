OpenCL-Testing
==============

Sandbox for OpenCL prototypes and development.

Setup
------

You Need OpenCL 1.2 to run and compile the the code in this repository. For Nvidia users, this should be done by installing the [cuda toolkit from nvidias website](https://developer.nvidia.com/cuda-downloads).

This code was tested using an Nvidia graphics card (GT555m) on a 64bit linux Machine with the latest Nvidia 331 drivers.

For those of you who are having difficulty setting up OpenCL on linux with optimus enabled graphics cards, see
[this post on stack overflow](http://stackoverflow.com/questions/20335579/error-clgetplatformids-1001-when-running-opencl-code-linux/) which should cover all the problems you have (which I had to deal with too). Feel free to comment on the post if you are still having trouble getting it to work.

Compiling
---------

You should use g++ using the C++11 specification:

    g++ -std=c++11 -I /usr/local/cuda-5.5 -L /usr/lib/nvidia-331 matrix_mul.cpp -o matrix_mul.exe -lOpenCL

Make sure to change the location of the OpenCL headers as necessary for your system. In my case I have cuda-5.5 installed and nvidia-331 drivers.

