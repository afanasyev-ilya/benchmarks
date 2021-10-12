#pragma once

enum DATATYPE_USED
{
    __FLOAT__ = 0,
    __DOUBLE__ = 1
};

enum OPT_MODE
{
    GENERIC = 0,
    OPTIMIZED = 1
};


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

    DATATYPE_USED datatype;
    OPT_MODE opt_mode;

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
    DATATYPE_USED get_datatype() {return datatype;};
    OPT_MODE get_opt_mode(){return opt_mode;};
    
    void parse_args(int _argc, char **_argv);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
