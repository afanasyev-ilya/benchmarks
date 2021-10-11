#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    int radius;
    int mode;

    int size;
    int deg;

    int small_size;
    int large_size;

    size_t length;
public:
    Parser();
    
    int get_radius() { return radius; };
    int get_mode() { return mode; };
    int get_size() { return size; };
    int get_deg() { return deg; };

    int get_small_size() {return small_size;};
    int get_large_size() {return large_size;};

    size_t get_length() { return length; };
    
    void parse_args(int _argc, char **_argv);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
