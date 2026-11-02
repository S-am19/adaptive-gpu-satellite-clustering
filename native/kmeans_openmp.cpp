// C++ OpenMP Lloyd baseline. Binary I/O; no high-level clustering library.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

template<class T> void read_value(T& v) { if(!std::cin.read(reinterpret_cast<char*>(&v),sizeof(v))) throw std::runtime_error("truncated input"); }
template<class T> void write_value(const T& v) { std::cout.write(reinterpret_cast<const char*>(&v),sizeof(v)); }
int main() {
  try {
    char magic[8]; if(!std::cin.read(magic,8)||std::memcmp(magic,"PCAPOMP1",8)) throw std::runtime_error("invalid protocol");
    uint64_t n,d,k,limit; double tolerance; uint32_t threads;
    read_value(n);read_value(d);read_value(k);read_value(limit);read_value(tolerance);read_value(threads);
    if(!n||!d||!k||k>n||!limit||!threads||!std::isfinite(tolerance)||tolerance<0||n>INT32_MAX) throw std::runtime_error("invalid shape/configuration");
    if(d>SIZE_MAX/n || d>SIZE_MAX/k || k*d>SIZE_MAX/threads) throw std::runtime_error("shape overflow");
    std::vector<float> X(n*d), C(k*d);
    if(!std::cin.read(reinterpret_cast<char*>(X.data()),X.size()*sizeof(float))||!std::cin.read(reinterpret_cast<char*>(C.data()),C.size()*sizeof(float))) throw std::runtime_error("truncated arrays");
    for(float x:X) if(!std::isfinite(x))throw std::runtime_error("nonfinite X");
    for(float c:C) if(!std::isfinite(c))throw std::runtime_error("nonfinite C0");
    omp_set_dynamic(0);omp_set_num_threads(threads);
    std::vector<int32_t> labels(n);
    std::vector<double> sums(threads*k*d), objective, movement;
    std::vector<uint64_t> counts(threads*k);
    auto start=std::chrono::steady_clock::now();
    auto assign=[&]() {
      double sse=0;
      #pragma omp parallel for schedule(static) reduction(+:sse)
      for(uint64_t i=0;i<n;++i) {
        double best=std::numeric_limits<double>::infinity();int32_t id=0;
        for(uint64_t j=0;j<k;++j) {
          double dist=0;
          for(uint64_t f=0;f<d;++f) {double delta=double(X[i*d+f])-double(C[j*d+f]);dist+=delta*delta;}
          if(dist<best){best=dist;id=int32_t(j);}
        }
        labels[i]=id;sse+=best;
      }
      return sse;
    };
    objective.push_back(assign());movement.push_back(0);
    uint64_t steps=0;uint32_t converged=0,actual_threads=0;double final_assignment_s=0;
    // Record the actual worker count, which can differ under runtime limits.
    #pragma omp parallel
    {
      #pragma omp single
      actual_threads=omp_get_num_threads();
    }
    for(uint64_t step=1;step<=limit;++step) {
      std::fill(sums.begin(),sums.end(),0);std::fill(counts.begin(),counts.end(),0);
      #pragma omp parallel
      {
        auto t=uint64_t(omp_get_thread_num());
        #pragma omp for schedule(static)
        for(uint64_t i=0;i<n;++i) {
          auto j=uint64_t(labels[i]);++counts[t*k+j];
          for(uint64_t f=0;f<d;++f)sums[(t*k+j)*d+f]+=double(X[i*d+f]);
        }
      }
      double max_shift=0;
      #pragma omp parallel for schedule(static) reduction(max:max_shift)
      for(uint64_t j=0;j<k;++j) {
        uint64_t count=0;for(uint64_t t=0;t<threads;++t)count+=counts[t*k+j];
        double shift=0;
        if(count)for(uint64_t f=0;f<d;++f) {
          double sum=0;for(uint64_t t=0;t<threads;++t)sum+=sums[(t*k+j)*d+f];
          float value=float(sum/double(count));double delta=double(value)-double(C[j*d+f]);shift+=delta*delta;C[j*d+f]=value;
        }
        max_shift=std::max(max_shift,shift);
      }
      steps=step;movement.push_back(std::sqrt(max_shift));
      auto assign_start=std::chrono::steady_clock::now();objective.push_back(assign());
      final_assignment_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-assign_start).count();
      if(!std::isfinite(objective.back()))throw std::runtime_error("nonfinite objective");
      if(movement.back()<=tolerance){converged=1;break;}
    }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout.write("PCAPOUT1",8);write_value(steps);write_value(converged);write_value(actual_threads);write_value(elapsed);write_value(final_assignment_s);
    std::cout.write(reinterpret_cast<char*>(C.data()),C.size()*sizeof(float));
    std::cout.write(reinterpret_cast<char*>(labels.data()),labels.size()*sizeof(int32_t));
    std::cout.write(reinterpret_cast<char*>(objective.data()),objective.size()*sizeof(double));
    std::cout.write(reinterpret_cast<char*>(movement.data()),movement.size()*sizeof(double));
    return std::cout ? 0:3;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
