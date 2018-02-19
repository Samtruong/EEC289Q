#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thrust/scan.h>
using namespace std;
int main(int argc, char const *argv[])
{
  // int array[10] = {0,1,0,0,1,1,1,0,0,1};
  // thrust::exclusive_scan(thrust::device,array,array + 10,array);
  for(int i = 0; i<10; i++){cout<<thrust::random<<endl;}

  return 0;
}
