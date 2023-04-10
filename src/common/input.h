#ifndef __INPUT_H__
#define __INPUT_H__

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>

#include "timers.h"

using input_t = std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>;

input_t read_input(char* filename) {
  CpuTimer timer("read_input");

  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cout << "Error: cannot open file " << filename << std::endl;
  }

  std::vector<uint32_t> A, B;
  std::unordered_map<uint32_t, uint32_t> mapper;

  uint32_t a, b;
  while((file >> a >> b)) {
    auto it = mapper.find(a);
    if (it == mapper.end()) {
      mapper.emplace(a, mapper.size());
      a = mapper.size() - 1;
    } else {
      a = it->second;
    }

    it = mapper.find(b);
    if (it == mapper.end()) {
      mapper.emplace(b, mapper.size());
      b = mapper.size() - 1;
    } else {
      b = it->second;
    }

    A.push_back(a);
    B.push_back(b);
  }

  return std::make_tuple(A, B, mapper.size());
}

#endif // __INPUT_H__