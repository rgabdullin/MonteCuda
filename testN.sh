time ./CPU_program 16384 && time ./GPU_program 16384 16384
time ./CPU_program 32768 && time ./GPU_program 16384 32768
time ./CPU_program 65536 && time ./GPU_program 16384 65536
time ./CPU_program 1048576 && time ./GPU_program 16384 1048576
time ./CPU_program 16777216 && time ./GPU_program 16384 16777216
time ./CPU_program 268435456 && time ./GPU_program 16384 268435456
time ./GPU_program 16384 4294967296
time ./GPU_program 16384 68719476736
time ./GPU_program 16384 274877906944

