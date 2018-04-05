#ifndef __ARRAY_H__
#define __ARRAY_H__

typedef float FLOAT;
typedef FLOAT(*function)(FLOAT);

// Класс для выделения памяти в Unified Memory
class Managed {
public:
	void *operator new(size_t len);
	void operator delete(void *ptr);
};

// Класс массива для подсчета среднего
class Array : public Managed{
public:
	int length;
	
	FLOAT *data;
	FLOAT *gpu_data;

	Array(int length);
	~Array();
	
	void SyncWithDevice();
	void SyncWithHost();
	// Параллельное вычисление среднего значения по массиву
	FLOAT Avg();
};
	
#endif

