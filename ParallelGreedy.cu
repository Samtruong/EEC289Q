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


__global__ void GraphGenerator(int* matrix,int* dimension, int* address, int* h_graph, int V)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
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

__global__ void DimensionGenerator(int* matrix, int* dimension, int* address, int V)
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

      num2 = j*V + curand(&state) % V;

      holder = result[num1];

      result[num1] = result[num2];

      result[num2] = holder;

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
    int other_vertex = h_graph[a + i];
    int other_color = result[startingAddress + other_vertex];
    // printf("Comparing %i with vertex: %i\n", curVertex, other_vertex);
    if (color == other_color)
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
result[startingAddress +curVertex] = color;
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
  // int length = dimension[V - 1] + address[V - 1]; //length of h_graph;

  //copy to shared memory:

  //for(int i = index; i < length; i+= stride){d_graph[i] = h_graph[i];}
// printf("3\n");

  /*for(int i = index; i < V; i+= stride)
  {
    d_dimension[i] = dimension[i];
    d_address[i] = address[i];
  }*/
  //end copy to shared memory
  // printf("4\n");


  for(int j = index; j < numVersion; j +=stride)
  {
    for(int k = 0; k < V; k++)
    {
      int curVertex = sequence[j*V+k];
      // printf("Current Vertex: %d\n", curVertex);
      a = address[curVertex]; //address of first neighbor
      d = dimension[curVertex];//number of neighbor
      Color(h_graph,j*V,curVertex, a, d, result);

      // int offset, degree, other;

     //  printf("Vertex %d\n", k);
     //  for (int i=0; i < V; i++)
     //  {
     //     offset = address[i];
     //     degree = dimension[i];
     //     for (int l=0; l < degree; l++)
     //     {
   	 // other = h_graph[offset+l];
   	 // if (result[i] == result[other] && result[i] != 0)
   	 //     printf("Vertex %d and %d have the same color %d\n", i, other, result[i]);
     //     }
     //  }
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
   int * h_graph;
   int * sequence;
   int * dimension;
   int * address;
   int * result; //Added
   int V;
   int numVersion;

   numVersion = 10;


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

   cudaMallocManaged(&sequence, sizeof(int) * V * numVersion);
   cudaMallocManaged(&dimension,sizeof(int)*V);
   cudaMallocManaged(&address,sizeof(int)*V);
   cudaMallocManaged(&result, sizeof(int) *V*numVersion);


   int finalSolution[V];

   DimensionGenerator<<<256,1024>>>(matrix,dimension,address,V);
   cudaDeviceSynchronize();
   thrust::exclusive_scan(thrust::host,dimension,&dimension[V],address);
   cudaMallocManaged(&h_graph,sizeof(int)* (dimension[V-1]+address[V-1]));

   GraphGenerator<<<256,1024>>>(matrix,dimension,address,h_graph,V);
   cudaDeviceSynchronize();

   // for(int i = 0; i < dimension[V-1] + address[V-1] ; i++){cout<<h_graph[i];}
   // cout<<endl;

   PermutationGenerator<<<1,500>>>(V,sequence,numVersion,V);
   cudaDeviceSynchronize();

   RandomizedParallelGreedy<<<512,1024>>>
   (h_graph, dimension, address, sequence, V, numVersion, result);


   cudaDeviceSynchronize();


   // cudaDeviceSynchronize();

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
    // int counter0 = 0;
    // printf("sequence:\n");
    // for (int i = 0; i < V*numVersion; i++)
    // {
    //  cout << sequence[i] << " ";
    //  if(i%V == V-1){cout<<"sequence no: "<<counter0++<<endl;}
    // }
   CountColors(V,V*numVersion,result);
   // int counter1 = 0;
   for(int i = 0; i < V*numVersion; i++)
   {
     if(i%V == V-1)
     {
       //cout<<counter1++<<endl;
       finalSolution[i%V] = result[i];
       if(!IsValidColoring(matrix,V,finalSolution)){cout<<"InValid Solution"<<endl;}
       // for (int j = 0; j < V; j++)
       //    printf("%i at index%i\n", finalSolution[j], j);
     }
     finalSolution[i%V] = result[i];
   }

   cudaFree(h_graph);
   cudaFree(dimension);
   cudaFree(sequence);
   cudaFree(address);
   cudaFree(result);
   cudaFree(matrix);
   return 0;
}
