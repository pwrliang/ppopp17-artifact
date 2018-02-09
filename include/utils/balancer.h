//
// Created by liang on 2/8/18.
//

#ifndef GROUTE_BALANCER_H
#define GROUTE_BALANCER_H

#include <stdint.h>

namespace groute {
    typedef uint32_t index_t;
    void balanced_alloctor(index_t elems_num, index_t *p_degree, index_t blocksPerGrid, index_t *p_o_lbounds,
                           index_t *p_o_ubounds) {
        int total_degree = 0;

        for (int v_idx = 0; v_idx < elems_num; v_idx++) {
            total_degree += p_degree[v_idx];
        }

        int avg_degree = total_degree / blocksPerGrid;
        assert(avg_degree > 0);

        int start_idx = 0;
        int end_idx = 0;
        int degree_in_block = 0;
        int bid = 0;
        for (int v_idx = 0; v_idx < elems_num; v_idx++) {
            degree_in_block += p_degree[v_idx];

            if (degree_in_block >= avg_degree) {
                end_idx = v_idx + 1;    // include this vertex

                end_idx = std::min<index_t>(end_idx, elems_num);
                p_o_lbounds[bid] = start_idx;
                p_o_ubounds[bid] = end_idx;
                bid++;
                start_idx = end_idx;
                degree_in_block = 0;
            }
        }
        p_o_lbounds[bid] = start_idx;
        p_o_ubounds[bid] = elems_num;
    }
}
#endif //GROUTE_BALANCER_H
