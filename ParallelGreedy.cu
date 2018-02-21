#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;


__global__ void CopyGraph(int* h_graph, int* pre_graph, int length)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < length; i += stride){h_graph[i] = pre_graph[i];}
}

__global__ void GraphGenerator(int* matrix, int* dimension, int* address, int V, int *h_graph)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < V; i += stride)
  {
    for (int j = 0; j < V; j++)
    {
      if(matrix[i*V + j])
      {
        dimension[i]++;
      }
    }
  }
  __syncthreads();
  thrust::exclusive_scan(thrust::device,&dimension[0],&dimension[V], &address[0]);
  for(int i = index; i < V; i += stride)
  {
    int a = address[i];
    int j = 0;
    for (int k = 0; k < V; k++)
    {
      if (matrix[i*V + k])
      {
        h_graph[a + j] = k;
        j++;
      }
    }
  }
}

__global__ void PermutationGenerator(int V, int*result, int numVersion, int shuffle_degree)
{
  unsigned long long seed = blockDim.x;
  unsigned long long sequence = threadIdx.x;
  unsigned long long offset = 0;
  curandState_t state;
  curand_init(seed,sequence,offset,&state);
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num1,num2,holder;
  for(int i = index; i < V*numVersion; i+= stride){result[i] = i%V;}
  __syncthreads();
  for(int j = index; j<numVersion; j+=stride)
  {
    for(int k = 0; k < shuffle_degree; k++)
    {
      num1 = j*V + curand(&state) % V;
	__syncthreads();
      num2 = j*V + curand(&state) % V;
	__syncthreads();
      holder = result[num1];
	__syncthreads();
      result[num1] = result[num2];
	__syncthreads();
      result[num2] = holder;
	__syncthreads();
    }

	//printf("for %i, num1 = %i, num2 = %i\n", j, num1%V, num2%V);
  }
}

__device__ void Color(int* h_graph, int startingAddress,int curVertex, int a, int d, int* result)
{
  //int result[curVertex] = 1;
  int color = 1;
  // printf("in color on vertex %i\n", curVertex);
  // //printf("h_graph\n");
  // for (int i = 0; i < d; i++)
  //   printf("%i ", h_graph[a+i]);
  // printf("\n");
  // printf("dimension %i\n", d);
  // printf("address %i\n", a);
  for (int i = 0; i < d; i++)
  {
    // printf("Comparing with vertex: %i\n", h_graph[a+i]);
    if (color == result[startingAddress + h_graph[a + i]])
    {
      // printf("clash with vertex %i\n", h_graph[a+i]);
      i = -1;
      color ++;
      //atomicAdd(&color, 1);
      continue;
    }
  }
  // printf("curVertex %i\n", curVertex);
  // printf("Vertex %i receieves color %i\n", curVertex,color);
__syncthreads();  
result[startingAddress +curVertex] = color;
__syncthreads();
}
__global__ void RandomizedParallelGreedy(int* h_graph, int* dimension,
                 int* address, int* sequence,int V, int numVersion, int* result)
{
  // printf("Sequence:\n");
  // for (int i = 0; i < V *numVersion; i++)
  // {
  //   printf("%i", sequence[i]);
  // }
  // printf("1\n");
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int a,d;

  // extern __shared__ int d_graph[];
  // extern __shared__ int d_dimension[];
  // extern __shared__ int d_address[];
  /*
  extern __shared__ int d_dimension[];
  extern __shared__ int d_address[];
  */
// printf("2\n");
  int length = dimension[V - 1] + address[V - 1]; //length of h_graph;

  //copy to shared memory:

  //for(int i = index; i < length; i+= stride){d_graph[i] = h_graph[i];}
  __syncthreads();
// printf("3\n");

  /*for(int i = index; i < V; i+= stride)
  {
    d_dimension[i] = dimension[i];
    d_address[i] = address[i];
  }*/
  __syncthreads();
  //end copy to shared memory
  // printf("4\n");

  for(int j = index; j < numVersion; j +=stride)
  {
    for(int k = 0; k < V; k++)
    {
      int curVertex = sequence[j*V+k];
      a = address[curVertex]; //address of first neighboor
      d = dimension[curVertex];//number of neighboor
      Color(h_graph,j*V,curVertex, a, d, result);
      __syncthreads();
    }
    // printf("nextVersion\n");
  }
}
//================================Utility Functions=======================================
void CountColors(int V,int length, int* color)
{
   int num_colors = 0;
   set<int> seen_colors;

   for (int i = 0; i < length; i++) {
      if (seen_colors.find(color[i]) == seen_colors.end())
      {
         seen_colors.insert(color[i]);
         num_colors++;
      }
      if(i%V==V-1)
      {
        cout<<num_colors<<endl;
        seen_colors.clear();
        num_colors = 0;
      }
   }
}
bool IsValidColoring(int* graph, int V, int* color)
{
   for (int i = 0; i < V; i++) {
      for (int j = 0; j < V; j++) {
         if (graph[i * V + j]) {
            if (i != j && color[i] == color[j]) {
               printf("Vertex %d and Vertex %d are connected and have the same color %d\n", i, j, color[i]);
               return false;
            }
            if (color[i] < 1) {
               printf("Vertex %d has invalid color %d\n", i, color[i]);
               return false;
            }
         }
      }
   }

   return true;
}
//Load raw .co data
void getDimension(const char filename[], int* V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   int num_rows;

   while (getline(infile, line))
   {
      istringstream iss(line);
      string s;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> num_rows;
         *V = num_rows;
         break;
      }
   }
   infile.close();
}

void ReadColFile(const char filename[], int* graph, int V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
      int node1, node2;
      iss >> s;
      if (s != "e")
         continue;

      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (graph)[(node1 - 1) * V + (node2 - 1)] = 1;
      (graph)[(node2 - 1) * V + (node1 - 1)] = 1;
   }
   infile.close();
}

//print graph Matrix
void PrintMatrix(int* matrix, int M, int N) {
   for (int row=0; row<M; row++)
   {
      for(int columns=0; columns<N; columns++)
      {
         printf("%i", matrix[row * N + columns]);
      }
      printf("\n");
   }
}


//===================================Main=======================================

int main(int argc, char* argv[])
{
   int * matrix;
   int * pre_graph;
   int * h_graph;
   int * sequence;
   int * dimension;
   int * address;
   int * result; //Added
   int V;
   int numVersion;

   numVersion = 1;


   if (string(argv[1]).find(".col") != string::npos)
   {
     getDimension(argv[1], &V);
     cudaMallocManaged(&matrix,sizeof(int)*V*V);
     ReadColFile(argv[1],matrix,V);
   }

   //else if (string(argv[1]).find(".mm") != string::npos)
      //ReadMMFile(argv[1], &graph, &V);
   else
      return -1;

/*
  int* testSequence;
  cudaMallocManaged(&testSequence,sizeof(int)*80);
  testSequence[0] = 0;
  testSequence[1] = 9;
  testSequence[2] = 33;
  testSequence[3] = 16;
  testSequence[4] = 35;
  testSequence[5] = 5;
  testSequence[6] = 15;
  testSequence[7] = 28;
  testSequence[8] = 8;
  testSequence[9] = 37;
  testSequence[10] = 36;
  testSequence[11] = 29;
  testSequence[12] = 20;
  testSequence[13] = 17;
  testSequence[14] = 3;
  testSequence[15] = 1;

  testSequence[16] = 13;
  testSequence[17] = 25;
  testSequence[18] = 18;
  testSequence[19] = 26;
  testSequence[20] = 32;
  testSequence[21] = 21;
  testSequence[22] = 19;
  testSequence[23] = 23;
  testSequence[24] = 30;
  testSequence[25] = 11;
  testSequence[26] = 31;
  testSequence[27] = 27;
  testSequence[28] = 24;
  testSequence[29] = 14;
  testSequence[30] = 6;
  testSequence[31] = 4;
  testSequence[32] = 38;
  testSequence[33] = 39;
  testSequence[34] = 34;
  testSequence[35] = 10;
  testSequence[36] = 2;
  testSequence[37] = 22;
  testSequence[38] = 7;
  testSequence[39] = 12;

for (int i = 40; i < 80; i++)
{
	testSequence[i] = testSequence[i-40];
}
*/
   cudaMallocManaged(&sequence, sizeof(int) * V * numVersion);
   cudaMallocManaged(&dimension,sizeof(int)*V);
   cudaMallocManaged(&address,sizeof(int)*V);
   cudaMallocManaged(&result, sizeof(int) *V*numVersion);
   cudaMallocManaged(&pre_graph,sizeof(int)*V*V);

   int finalSolution[V];
   GraphGenerator<<<256,1024>>>(matrix,dimension,address,V,pre_graph);
   cudaDeviceSynchronize();
   cudaMallocManaged(&h_graph,sizeof(int)* (dimension[V-1]+address[V-1]));
   CopyGraph<<<256,1024>>>(h_graph,pre_graph,dimension[V-1]+address[V-1]);
   cudaDeviceSynchronize();

   PermutationGenerator<<<1,500>>>(V,sequence,numVersion,V);
   cudaDeviceSynchronize();

   RandomizedParallelGreedy<<<1,1024>>>
   (h_graph, dimension, address, sequence, V, numVersion, result);
   cudaDeviceSynchronize();

   //
   // printf("dimensions\n");
   // for (int i = 0; i < V; i++)
   // {
   //   cout << dimension[i] << " ";
   // }
   // cout << endl;
   // printf("address\n");
   // for (int i = 0; i < V; i++)
   // {
   //   cout << address[i] << " ";
   // } cout << endl;
   // for (int i =0; i < (dimension[V-1]+address[V-1]); i++){printf("%i ", h_graph[i]);}
   //
   // cout<<endl;
   // printf("coloring:\n");
   // for (int i = 0; i < V*numVersion; i++)
   // {
   //  cout << result[i] << " ";
   //  if(i%V == V-1){cout<<endl;}
   // }
   //
    int counter0 = 0;
    printf("sequence:\n");
    for (int i = 0; i < V*numVersion; i++)
    {
     cout << sequence[i] << " ";
     if(i%V == V-1){cout<<"sequence no: "<<counter0++<<endl;}
    }
   CountColors(V,V*numVersion,result);
   int counter1 = 0;
   for(int i = 0; i < V*numVersion; i++)
   {
     if(i%V == V-1)
     {
       cout<<counter1++<<endl;
       finalSolution[i%V] = result[i];
       if(IsValidColoring(matrix,V,finalSolution)){cout<<"Valid Solution"<<endl;}
     }
     finalSolution[i%V] = result[i];
   }
   cudaFree(h_graph);
   cudaFree(dimension);
   cudaFree(sequence);
   cudaFree(address);
   cudaFree(result);
   cudaFree(matrix);
   cudaFree(pre_graph);
   return 0;
}
