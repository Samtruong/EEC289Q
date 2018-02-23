#include "ParallelGreedy.cu"
int main(int argc, char const *argv[]) {
  int** color = (int**) malloc(1*sizeof(int*));
  GraphColoringGPU(argv[1],color);
  return 0;
}
