//
// Created by liang on 2/6/18.
//

#ifndef GROUTE_READRANKS_H
#define GROUTE_READRANKS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

void ReadRanks(const char *path, std::vector<float> *ranks) {
    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "file " << path << " does not exists"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int line_no = 0;

    while (getline(infile, line)) {
        line_no++;

        if (line_no <= 2)
            continue;

        std::vector<std::string> tmp_res;
        boost::split(tmp_res, line, boost::is_any_of("\t "), boost::token_compress_on);
        if(tmp_res.size()==3) {
            int node = std::stoi(tmp_res[1]);
            float rank = std::stof(tmp_res[2]);
            ranks->at(node) = rank;
        }
    }
    infile.close();
}

#endif //GROUTE_READRANKS_H
