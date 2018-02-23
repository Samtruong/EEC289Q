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

  int color = 1;
  for (int i = 0; i < d; i++)
  {
    int other_vertex = h_graph[a + i];
    int other_color = result[startingAddress + other_vertex];
    if (color == other_color)
    {
      i = -1;
      color ++;
      continue;
    }
  }
result[startingAddress +curVertex] = color;
}

__global__ void RandomizedParallelGreedy(int* h_graph, int* dimension,
                 int* address, int* sequence,int V, int numVersion, int* result)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int a,d;

  for(int j = index; j < numVersion; j +=stride)
  {
    for(int k = 0; k < V; k++)
    {
      int curVertex = sequence[j*V+k];
      a = address[curVertex]; //address of first neighbor
      d = dimension[curVertex];//number of neighbor
      Color(h_graph,j*V,curVertex, a, d, result);
    }
  }
}
//================================Utility Functions=======================================
void CountColors(int V,int length, int* color, int &minColors, int &minIndex)
{
	//int minColors = INT_MAX;
	//int minIndex;
   int *num_colors;
	num_colors = (int*) malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
	{
		num_colors[i] = 0;
	}
   set<int> seen_colors;

   for (int i = 0; i < length; i++) {
      if (seen_colors.find(color[i]) == seen_colors.end())
      {
         seen_colors.insert(color[i]);
         num_colors[i/V]++;
      }
      if(i%V==V-1)
      {
        //cout<<num_colors[i/V]<<endl;
	if (num_colors[i/V] < minColors)
	{
		minColors = num_colors[i/V];
		minIndex = i / V;
	}
        seen_colors.clear();
        //num_colors = 0;
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


// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
void ReadMMFile(const char filename[], bool** graph, int* V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   // Reading comments
   while (getline(infile, line)) {
      istringstream iss(line);
      if (line.find('%') == string::npos)
         break;
   }

   // Reading metadata
   istringstream iss(line);
   int num_rows, num_cols, num_edges;
   iss >> num_rows >> num_cols >> num_edges;

   *graph = new bool[num_rows * num_rows];
   memset(*graph, 0, num_rows * num_rows * sizeof(bool));
   *V = num_rows;

   // Reading nodes
   while (getline(infile, line)) {
      istringstream iss(line);
      int node1, node2, weight;
      iss >> node1 >> node2 >> weight;

      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = true;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = true;
   }
   infile.close();
}


//===================================Main=======================================

int main(int argc, char* argv[])
{
   int * matrix;
   int * h_graph;
   int * sequence;
   int * dimension;
   int * address;
   int * result;
   int V;
   int numVersion;

   //numVersion = 500;


   if (string(argv[1]).find(".col") != string::npos)
   {
     getDimension(argv[1], &V);
     cudaError_t result = cudaMallocManaged(&matrix,sizeof(int)*V*V);
     const char *error = cudaGetErrorString(result);
     printf("%s\n", error);
     ReadColFile(argv[1],matrix,V);
   }
   /*
   else if (string(argv[1]).find(".mm") != string::npos)
      ReadMMFile(argv[1], matrix, V);*/
   else
      return -1;

	numVersion = 1000;
   cudaMallocManaged(&sequence, sizeof(int) * V * numVersion);
   cudaMallocManaged(&dimension,sizeof(int)*V);
   cudaMallocManaged(&address,sizeof(int)*V);
   cudaMallocManaged(&result, sizeof(int) *V*numVersion);


   DimensionGenerator<<<256,1024>>>(matrix,dimension,address,V);
   cudaDeviceSynchronize();
   thrust::exclusive_scan(thrust::host,dimension,&dimension[V],address);
   cudaMallocManaged(&h_graph,sizeof(int)* (dimension[V-1]+address[V-1]));

   GraphGenerator<<<256,1024>>>(matrix,dimension,address,h_graph,V);
   cudaDeviceSynchronize();

   PermutationGenerator<<<1, numVersion>>>(V,sequence,numVersion,V);
   cudaDeviceSynchronize();

   RandomizedParallelGreedy<<<512,1024>>>
   (h_graph, dimension, address, sequence, V, numVersion, result);


   cudaDeviceSynchronize();

	int numColors = INT_MAX;
	int minIndex = 0;
   CountColors(V,V*numVersion,result, numColors, minIndex);
   
int finalSolution[V];
   for(int i = 0; i < V*numVersion; i++)
   {
     if(i%V == V-1)
     {
       finalSolution[i%V] = result[i];
       if(!IsValidColoring(matrix,V,finalSolution)){cout<<"InValid Solution"<<endl;}
     }
     finalSolution[i%V] = result[i];
   }

	
/*	cout << "Final Coloring" << endl;
	for (int i = 0; i < V; i++)
		cout << result[minIndex*V+i] << " ";
*/	//cout << "Number of colors: " << numColors << endl;
	//cout << IsValidColoring(matrix, V, result + minIndex*V) << endl;

 *color = result;
  
 cudaFree(h_graph);
   cudaFree(dimension);
   cudaFree(sequence);
   cudaFree(address);
   cudaFree(matrix);
   return 0;
}
