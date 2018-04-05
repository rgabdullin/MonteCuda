all: CPU_program GPU_program

CPU_program:
	g++ -std=c++11 -o CPU_program MonteCarloCPU.cpp -O3

GPU_program: ./CUDA/Array.o ./CUDA/MonteCarloGPU.o
	nvcc -std=c++11 -arch=sm_35 -rdc=true -o GPU_program ./CUDA/Array.o ./CUDA/MonteCarloGPU.o -O3

./CUDA/Array.o:
	nvcc -std=c++11 -arch=sm_35 -rdc=true -o ./CUDA/Array.o -c ./CUDA/Array.cu -O3

./CUDA/MonteCarloGPU.o:
	nvcc -std=c++11 -arch=sm_35 -rdc=true -o ./CUDA/MonteCarloGPU.o -c ./CUDA/MonteCarloGPU.cu -O3

clean:
	rm -f CPU_program GPU_program ./CUDA/Array.o ./CUDA/MonteCarloGPU.o
