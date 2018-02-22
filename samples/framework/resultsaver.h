//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_RESULTSAVER_H
#define GROUTE_RESULTSAVER_H

#include <groute/graphs/common.h>
#include <glog/logging.h>

DECLARE_int32(top_values);

template<typename TValue>
bool ResultOutput(const char *file, const std::vector<TValue> &values, bool sort) {
    FILE *f;
    f = fopen(file, "w");

    if (!f) {
        LOG(WARNING) << "Could not open '" << file << "' for writing\n";
        return false;
    }

    int output_num = FLAGS_top_values;

    if (output_num == -1) {
        LOG(WARNING) << "WARN:output all values";
        output_num = values.size();
    }

    VLOG(0) << "Writing to file ...";
    fprintf(f, "VALUES 1--%d of %d\n", FLAGS_top_values, (int) values.size());

    if (sort) {
        struct node_value {
            index_t node;
            TValue value;

            inline bool operator<(const node_value &rhs) const {
                return value < rhs.value;
            }
        } *p_node_value;

        p_node_value = (struct node_value *) calloc(values.size(), sizeof(struct node_value));

        if (!p_node_value) {
            LOG(WARNING) << "ResultOutput: Failed to allocate memory!";
            return false;
        }

        for (int i = 0; i < values.size(); i++) {
            p_node_value[i].node = i;
            p_node_value[i].value = values[i];
        }

        VLOG(0) << "Sorting by value ...";
        std::stable_sort(p_node_value, p_node_value + values.size());

        for (int i = 1; i <= output_num; i++) {
            if (typeid(TValue) == typeid(float) || typeid(TValue) == typeid(double))
                fprintf(f, "%d %d %*e\n", i, p_node_value[values.size() - i].node, FLT_DIG, p_node_value[values.size() - i].value);
            else if (typeid(TValue) == typeid(short) || typeid(TValue) == typeid(int) || typeid(TValue) == typeid(long))
                fprintf(f, "%d %d %ld\n", i, p_node_value[values.size() - i].node, p_node_value[values.size() - i].value);
            else if (typeid(TValue) == typeid(unsigned) || typeid(TValue) == typeid(unsigned int) || typeid(TValue) == typeid(unsigned long))
                fprintf(f, "%d %d %uld\n", i, p_node_value[values.size() - i].node, p_node_value[values.size() - i].value);
        }
        free(p_node_value);
    } else {
        for (int i = 0; i < output_num; i++) {
            if (typeid(TValue) == typeid(float) || typeid(TValue) == typeid(double))
                fprintf(f, "%d %*e\n", i, FLT_DIG, values[i]);
            else if (typeid(TValue) == typeid(short) || typeid(TValue) == typeid(int) || typeid(TValue) == typeid(long))
                fprintf(f, "%d %ld\n", i, values[i]);
            else if (typeid(TValue) == typeid(unsigned) || typeid(TValue) == typeid(unsigned int) || typeid(TValue) == typeid(unsigned long))
                fprintf(f, "%d %uld\n", i, values[i]);
        }
    }

    fclose(f);

    return true;
}

#endif //GROUTE_RESULTSAVER_H
