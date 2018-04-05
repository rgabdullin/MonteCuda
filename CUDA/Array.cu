#ifndef SYNC_AND_CHECK_CUDA_ERRORS
#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); fclose(stdout);exit(1); }}
#endif

#include "Array.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <cstdlib>
#include <cstdio>

#define BLOCKSIZE 256

// Выделение памяти в Unified Memory для объекта
void* Managed::operator new(size_t len) {
	void *ptr;
	cudaMallocManaged(&ptr, len);
	SYNC_AND_CHECK_CUDA_ERRORS;
	return ptr;
}

// Удаление объекта в Unified Memory
void Managed::operator delete(void *ptr) {
	SYNC_AND_CHECK_CUDA_ERRORS;
	cudaFree(ptr);
}

// Конструктор
Array::Array(int l_length) {
	length = l_length;
	
	
	data = (FLOAT*)malloc(sizeof(FLOAT) * length);
	
	// Выделение памяти на GPU
	cudaMalloc((void**)&gpu_data, sizeof(FLOAT) * length);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

// Деструктор
Array::~Array() {
	free(data);

	// Освобождение памяти
	SYNC_AND_CHECK_CUDA_ERRORS;
	cudaFree(gpu_data);
}

// Копирование массива с GPU на CPU
void Array::SyncWithDevice() {
	SYNC_AND_CHECK_CUDA_ERRORS;
	cudaMemcpy(data, gpu_data, sizeof(FLOAT) * length, cudaMemcpyDeviceToHost);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

// Копирование массива с CPU на GPU
void Array::SyncWithHost() {
	SYNC_AND_CHECK_CUDA_ERRORS;
	cudaMemcpy(gpu_data, data, sizeof(FLOAT) * length, cudaMemcpyHostToDevice);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

// Суммирование массива деревом с заданным весом элементов
__global__
void Reduce(int num, FLOAT* in, FLOAT* out, FLOAT weight = 1) {
	// Создание буффера в shared memory 
	__shared__ FLOAT data[BLOCKSIZE];
	// Номер нити в блоке
	int tid = threadIdx.x;
	// Номер блока
	int bid = gridDim.x * blockIdx.y + blockIdx.x;
	// Номер элемента входного массива, соответствующий данной нити
	int idx = bid * blockDim.x + tid;

	// Проверка, чтобы не суммировать лишнее
	if(idx < num){
		data[tid] = weight * in[idx];
		__syncthreads();
	}
	__syncthreads();
	
	// Суммирование деревом
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		// Проверка - нужно ли суммировать данный элемент на итерации с шагом s
		if (tid < s) {
			// Проверка, чтобы не суммировать лишнее
			if (idx + s < num)
				data[tid] = data[tid] + data[tid + s];
		}
		__syncthreads();
	}
	
	// Запись
	if (tid == 0) {
		out[bid] = data[0];
	}
	__syncthreads();
}

// Прототип функции для квадратной сетки блоков
dim3 squareGrid(int block_num);

// Поиск среднего значения по массиву
FLOAT Array::Avg() {
	// Заводится два буфера 
	FLOAT* buffer[2]; int len[2];
	
	// Выделение памяти под первый буффер
	len[0] = length / BLOCKSIZE + (length % BLOCKSIZE ? 1 : 0);
	cudaMalloc((void**)&buffer[0], sizeof(FLOAT) * len[0]);
	SYNC_AND_CHECK_CUDA_ERRORS;
	
	// Первое суммирование length элементов по блокам размера BLOCKSIZE в новый массив из length / BLOCKSIZE элеметов
	Reduce <<<squareGrid(len[0]), BLOCKSIZE>>> (length, gpu_data, buffer[0]);
	SYNC_AND_CHECK_CUDA_ERRORS;

	// Если остался один элемент, то возвращаем ответ
	if (len[0] == 1) {
		FLOAT answer;
		cudaMemcpy(&answer, buffer[0], sizeof(FLOAT), cudaMemcpyDeviceToHost);
		SYNC_AND_CHECK_CUDA_ERRORS;

		cudaFree(buffer[0]); SYNC_AND_CHECK_CUDA_ERRORS;
		return answer / length;
	}
	// Иначе продолжаем суммирование используя 2 буффера поочередно
	// Выделяем память под второй буффер
	len[1] = len[0] / BLOCKSIZE + (len[0] % BLOCKSIZE ? 1 : 0);
	cudaMalloc((void**)&buffer[1], sizeof(FLOAT) * len[1]);
	SYNC_AND_CHECK_CUDA_ERRORS;
	
	// До тех пор, пока не останется 1 число
	do {
		// Суммируем
		len[1] = len[0] / BLOCKSIZE + (len[0] % BLOCKSIZE ? 1 : 0);
		Reduce <<<squareGrid(len[1]), BLOCKSIZE>>> (len[0], buffer[0], buffer[1]);
		SYNC_AND_CHECK_CUDA_ERRORS;
		len[0] = len[1];

		// Меняем местами буфферы
		FLOAT* _tmp = buffer[0];
		buffer[0] = buffer[1];
		buffer[1] = _tmp;
	} while (len[1] != 1);
	
	// Возврат ответа
	FLOAT answer;
	cudaMemcpy(&answer, buffer[0], sizeof(FLOAT), cudaMemcpyDeviceToHost);
	SYNC_AND_CHECK_CUDA_ERRORS;

	cudaFree(buffer[0]); cudaFree(buffer[1]); SYNC_AND_CHECK_CUDA_ERRORS;
	return answer / length;
}
