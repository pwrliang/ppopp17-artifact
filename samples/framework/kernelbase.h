//
// Created by liang on 2/5/18.
//

#ifndef GROUTE_KERNELBASE_H
#define GROUTE_KERNELBASE_H

#include <groute/graphs/common.h>

namespace maiter {
    template<class V, class D>
    struct IterateKernel {
//        virtual void read_data(string& line, K& k, D& data) = 0;
        __device__ virtual V InitValue(const index_t node, unsigned out_degree) = 0;
        __device__ virtual D InitDelta(const index_t node, unsigned out_degree) = 0;


        virtual const V &default_v() const = 0;

        virtual void init_v(const K &k, V &v, D &data) = 0;

        virtual void accumulate(V &a, const V &b) = 0;

        virtual void priority(V &pri, const V &value, const V &delta) = 0;

        virtual void g_func(const K &k, const V &delta, const V &value, const D &data, vector <pair<K, V>> *output) = 0;
    };

    struct Algo {
        static const char *Name() { return "Maiter Kernel"; }
    };


}

#endif //GROUTE_KERNELBASE_H
