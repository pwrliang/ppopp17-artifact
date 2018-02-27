//
// Created by liang on 2/26/18.
//
#include <utils/graphs/traversal.h>
#include "pr_common.h"

namespace dual{
    struct Algo {
        static const char *NameLower() { return "dual"; }

        static const char *Name() { return "DUAL"; }

    };
}

bool DualGPU(){
//    int ngpus = 2;
//    utils::traversal::Context<dual::Algo> context(ngpus);
//    context.PrintStatus();
//    groute::graphs::multi::CSRGraphAllocator dev_graph_allocator(context,context.host_graph,ngpus,false);
//
//    groute::graphs::multi::NodeOutputLocalDatum<rank_t> current_ranks;
//    groute::graphs::multi::NodeOutputGlobalDatum<rank_t> residual;
//
//    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual);

}