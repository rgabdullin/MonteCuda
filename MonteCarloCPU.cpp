#include <random>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>

typedef float FLOAT;
typedef FLOAT(*function)(FLOAT);

FLOAT func(FLOAT x) {
	return (FLOAT) (log(1 + x) * sin(x));
}

FLOAT ComputeIntegral(function func, FLOAT a, FLOAT b, long count_of_numbers) {
	FLOAT integral = 0, loc_int = 0;
	int threshold = (int)sqrt(count_of_numbers);

	int seed = time(0);

	srand(seed);

	for (unsigned long i = 0, k = 0; i < count_of_numbers; ++i, ++k) {
		FLOAT x = a + (b-a) * ((FLOAT)rand()) / RAND_MAX;
		loc_int += func(x);
		if (k > threshold) {
			integral += loc_int / count_of_numbers;
			loc_int = 0;
			k = 0;
		}
	}

	integral += loc_int / count_of_numbers;

	return (b - a) * integral;
}

int main(int argc, char * argv[]) {
	long count_of_numbers = 1024 * 1024 * 64;
	
	if(argc > 1)
		sscanf(argv[1],"%ld",&count_of_numbers);
	
	printf("Computing %lu points with 1 CPU thread\n", count_of_numbers); 
	
	FLOAT int_func = 0;	
	int_func = ComputeIntegral(func, 0, 0.5 * M_PI, count_of_numbers);
	
	printf("\nint[0;pi/2] f(x) dx = %.8lf\n", int_func);
    return 0;
}

