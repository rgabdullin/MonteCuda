#ifndef SYNC_AND_CHECK_CUDA_ERRORS
#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); fclose(stdout); exit(1); }}
#endif

#include "Array.h"

#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand_kernel.h"
#include "curand.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define BLOCKSIZE 256

// Функция, которая по количеству блоков возвращает прямоугольную сетку, максимально близкую к квадратной
dim3 squareGrid(int block_num) {
	int a = sqrt(block_num);
	while(block_num % a)
		a--;
	return dim3(a, block_num / a);
}

// Ядро, которое генерирует выборку N случайных точек и возвращает среднее значение функции по выборке
__global__
void ComputeLocalAvgs(Array* grid, function func, FLOAT a, FLOAT b, int seed, long count_of_numbers) {
	// Узнаем номер нити - нужно для записи результата в выходной массив
	int idx = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	
	// Состояние для генератора псевдослучайных чисел.
	curandState_t state;

	// Инициализация генератора с зерном seed, выбор idx-той подпоследовательности 
	curand_init(seed, idx, 0, &state);
	// Синхронизация нитей
	__syncthreads();

	FLOAT integral = 0, loc_int = 0;
	
	// Создаем в разделяемой памяти переменную threshold и загружаем ее нулевой нитью блока
	__shared__ FLOAT threshold; if(threadIdx.x == 0) threshold = sqrtf(count_of_numbers);
	__syncthreads();

	// Начинаем считать среднее значение
	for (unsigned long i = 0; i < count_of_numbers; ++i) {
		// Генерируем очередное равномерное случайное значение с отрезка интегрирования
		FLOAT x = a + (b - a) * curand_uniform(&state);
		
		// Суммируем в локальную сумму
		loc_int += func(x);
		// Когда набрана достаточная сумма - добавляем ее в среднее значение.
		// Это для того, чтобы не было проблем с точностью float и double при больших размерах выборки.
		if (fabs(loc_int) > threshold) {
			integral += loc_int / count_of_numbers;
			loc_int = 0;
		}
		// Синхронизация
		__syncthreads();	
	}

	// Добавляем в среднее значение оставшуюся сумму
	integral += loc_int / count_of_numbers;

	// Заносим ср. значение в выходной массив
	grid->gpu_data[idx] = integral * (b - a);
}

// Функция интегрирования
FLOAT ComputeIntegral_GPU(long number_of_threads, function func, FLOAT a, FLOAT b, int seed, long count_of_numbers) {
	FLOAT integral = 0;
	
	// Если количество нитей не делится на размер блока, добиваем в большую сторону до кратности
	// Для упрощения алгоритма. Иначе будет потеря в производительности
	if (number_of_threads % BLOCKSIZE) {
		number_of_threads = (number_of_threads / BLOCKSIZE + 1) * BLOCKSIZE;
		printf("warning::number_of_threads %% BLOCKSIZE != 0, using closest: number_of_threads = %ld\n", number_of_threads); fflush(stdout);
	}
	
	// Аналогично, если размер выборки не делится на число нитей
	if (count_of_numbers % number_of_threads) {
		count_of_numbers = (count_of_numbers / number_of_threads + 1) * number_of_threads;
		printf("warning::count_of_numbers %% number_of_threads != 0, using closest: count_of_numbers = %ld\n", count_of_numbers); fflush(stdout);
	}
	
	// Создаем массив для подсчета M средних значений по выборкам из N / M с.в. 
	Array * Grid = new Array(number_of_threads);

	// Выбираем размеры блока и сетки
	dim3 grid = squareGrid(number_of_threads / BLOCKSIZE);
	dim3 block = dim3(BLOCKSIZE);

	printf("Computing sums of %ld numbers per each thread of %ld\n", count_of_numbers / number_of_threads, number_of_threads); fflush(stdout);
	// Запускаем вычисление 
	ComputeLocalAvgs <<<grid, block>>> (Grid, func, a, b, seed, count_of_numbers / number_of_threads);
	SYNC_AND_CHECK_CUDA_ERRORS;

	// Суммируем полученные ср.значения
	integral = Grid->Avg();
	SYNC_AND_CHECK_CUDA_ERRORS;
	
	delete Grid;
	
	// Возврат ответа
	return integral;
}

__device__
FLOAT func1(FLOAT x) {
	return (FLOAT)(logf(1 + x) * sin(x));
}

__device__ function d_pfunc = func1;

// Функция печати информации о видеокартах
void printInfo(void) {
	time_t start_time; time(&start_time);
	struct tm* s_time = localtime(&start_time);
	printf("Runnning. time = %04d/%02d/%02d %02d:%02d:%02d\n",
			s_time->tm_year + 1900,
			s_time->tm_mon + 1,
			s_time->tm_mday,
			s_time->tm_hour,
			s_time->tm_min,
			s_time->tm_sec);

	cudaDeviceProp props;
	int dev_count;

	cudaGetDeviceCount(&dev_count);

	printf("Detected %d devices:\n", dev_count);
	for (int i = 0; i < dev_count; ++i) {
		cudaGetDeviceProperties(&props, i);
		printf("\t[ %d ] %s, %.1f GBs memory, CUDA %d.%d Compute Capability\n", i, props.name, float(props.totalGlobalMem) / (1024 * 1024 * 1024), props.major, props.minor);
	}
	printf("\n");
	SYNC_AND_CHECK_CUDA_ERRORS;
}

int main(int argc, char* argv[]) {
	// Печать инфы про видеокарты 
	printInfo();

	long count_of_numbers = 1024 * 1024 * 64;
	int count_of_threads = 16 * 1024;
	
	// Первым параметром число нитей
	if(argc > 1)
		sscanf(argv[1],"%d",&count_of_threads);
	
	// Вторым параметром размер выборки
	if(argc > 2)
		sscanf(argv[2],"%ld",&count_of_numbers);
	
	printf("Computing %lu points with %d GPU threads\n", count_of_numbers, count_of_threads); 
	
	FLOAT int_func_gpu = 0;
	
	// Вытаскиваем указатель на функцию с памяти GPU
	function pfunc;
	cudaMemcpyFromSymbol(&pfunc, d_pfunc, sizeof(function));
	SYNC_AND_CHECK_CUDA_ERRORS;
	
	// Вычисляем интеграл
	int_func_gpu = ComputeIntegral_GPU(count_of_threads, pfunc, 0, 0.5 * M_PI, time(0), count_of_numbers);
	SYNC_AND_CHECK_CUDA_ERRORS;
	
	// Печатаем ответ
	printf("\nint[0;pi/2] f(x) dx = %.8lf\n", int_func_gpu);
    return 0;
}

