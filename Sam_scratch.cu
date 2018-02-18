#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thrust/scan.h>

int int main(int argc, char const *argv[]) {
  int array[6] = {0,1,0,0,1,1,1,0,0,1};
  thrust::exclusive_scan(array,array + 10,array)
  return 0;
}
