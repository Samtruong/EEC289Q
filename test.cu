#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <queue>

using namespace std;

//#include <device_vector.h>

  //cudaMallocManaged(& bins, numC*numV*sizeof(int));

/*
__global__
void makeBins(int * coloredGraph, int numV, int numC, std::queue<int>* bins)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index ; i < numV; i+= stride)
  {
    bins[coloredGraph[i]].push(i);
  }
}
*/

struct AdjListNode {
	int vertex;
	struct AdjListNode * next;
};

struct AdjList {
	struct AdjListNode * head;
};

struct AdjListNode* newAdjListNode(int vertex)
{
	struct AdjListNode * newNode = (struct AdjListNode * ) malloc(sizeof(struct AdjListNode));
	newNode -> vertex = vertex;
	newNode -> next = NULL;
	return newNode;
};

void populateList(bool *graph, int numV)
{
	struct AdjList list[numV];
	for (int i = 0; i < numV; i++)
	{
		for (int j = 0; j < numV; i++)
		{
			if (graph[i*numV + j])
			{
				struct AdjListNode* ptr = list[i].head;
				while (ptr -> next)
					ptr = ptr -> next;
				ptr -> next = newAdjListNode(j);
			}

		}

	}
}
/*
__global__
void reduceColors (bool *graph, int *coloredGraph, int numV, int numC, int numIterations, struct AdjList *list)
{
	for (int i = 0; i < numIterations; i++)
	{
		int vertex1 = rand();
		int vertex2 = rand();
		int vertex1Color = coloredGraph[vertex1];
		struct AdjListNode *ptr = list[vertex2].head;
		if (coloredGraph[vertex1] == coloredGraph[vertex2])
			continue;
		if (!graph[vertex1*numV + vertex2])
		{
			//loop through all adjacent vertices of vertex 2 to determine if same color exists.
			while (ptr -> next)
			{
				ptr = ptr -> next;
				if (coloredGraph[ptr -> vertex] == vertex1Color) continue;
			}	
			if (coloredGraph[vertex1] < coloredGraph[vertex2])
				coloredGraph[vertex2] = coloredGraph[vertex1];
			else
				coloredGraph[vertex1] = coloredGraph[vertex2];
		}
	}
}
*/

/*
__global__
void reduceBins(int * coloredGraph, int numV, int numC, std::queue<int> * bins, int threshold)
{
	int binIndex = blockIdx.x % numC;
	std::queue<int> bin = bins[binIndex];
	int iteration = 0;
	while (iteration < threshold)
	{	
		int currentVertex = queue.pop();
			

	}	

}
*/

void ReadColFile(const char filename[], bool** graph, int* V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   int num_rows, num_edges;

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
      int node1, node2;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> num_rows;
         iss >> num_edges;
         *V = num_rows;
         *graph = new bool[num_rows * num_rows];
         memset(*graph, 0, num_rows * num_rows * sizeof(bool));
         continue;
      } else if (s != "e")
         continue;

      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = true;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = true;
   }
   infile.close();
}



int main(int argc, char *argv[])
{

  bool *graph;
  int V;
  int *color;

   if (string(argv[1]).find(".col") != string::npos)
      ReadColFile(argv[1], &graph, &V);
  else
  	return -1;
//Code to make random graph
/*
  const int numV = 10;
  const int numC = 5;
  int coloredGraph[numV];

  for(int i = 0; i <  numV; i++)
  {
    coloredGraph[i] = rand() % 1000 + 1;
  }
*/


  
  //std::queue<int>  bins [numC];

  //makeBins<<<1,1>>>(coloredGraph, numV, numC, bins);
  
//cudaFree(bins);  
return 0;
}
