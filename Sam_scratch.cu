#include <unistd.h>
#include <stdio.h>
#include <iostream>
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

__global__ void PermutationGenerator(int V, int*result, int numVersion, int shuffle_degree)
{
  unsigned long long seed = blockDim.x;
  unsigned long long sequence = threadIdx.x;
  unsigned long long offset = 0;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num1,num2,holder;
  for(int i = index; i < V*numVersion; i+= stride){result[i] = i%V;}
  __syncthreads();
  curandState_t state;
  curand_init(seed,sequence,offset,&state);
  for(int j = index; j<numVersion; j+=stride)
  {
    for(int k = 0; k < shuffle_degree; k++)
    {
      num1 = j*V + curand(&state) % V;
      num2 = j*V + curand(&state) % V;
      holder = result[num1];
      result[num1] = result[num2];
      result[num2] = holder;
    }
  }
}
int main( )
{
  int* result;
  cudaMallocManaged(&result,sizeof(int)*100);
  PermutationGenerator<<<2,5>>>(5,result,20,5);
  cudaDeviceSynchronize();
  for(int i = 0; i < 100; i++){cout<<result[i];if(i % 5 == 4){cout<<endl;}  }
  cudaFree(result);
  return 0;
}
