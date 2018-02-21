//
// Created by liang on 2/20/18.
//

#ifndef GROUTE_GRAPH_API_H
#define GROUTE_GRAPH_API_H
namespace gframe {
    namespace api {
        struct GraphAPIBase {
            typedef struct {
                index_t nnodes;
                index_t nedges;
            }GraphInfo ;

            GraphInfo graphInfo;
        };
    }
}
#endif //GROUTE_GRAPH_API_H
