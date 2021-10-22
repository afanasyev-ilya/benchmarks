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

class ParserBenchmark
{
private:
    size_t radius;
    int mode;

    size_t size;
    int deg;

    size_t small_size;
    size_t large_size;

    DATATYPE_USED datatype;
    OPT_MODE opt_mode;

    int graph_scale;
public:
    ParserBenchmark();

    size_t get_radius() { return radius; };
    size_t get_mode() { return mode; };
    size_t get_size() { return size; };
    int get_deg() { return deg; };
    int get_graph_scale() {return graph_scale; };

    size_t get_small_size() {return small_size;};
    size_t get_large_size() {return large_size;};

    DATATYPE_USED get_datatype() {return datatype;};
    OPT_MODE get_opt_mode(){return opt_mode;};
    
    void parse_args(int _argc, char **_argv);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
