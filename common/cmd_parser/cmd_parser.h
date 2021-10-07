#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    int radius;

    size_t length;
public:
    Parser();
    
    int get_radius() { return radius; };
    size_t get_length() { return length; };
    
    void parse_args(int _argc, char **_argv);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
