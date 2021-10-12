#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    radius = 1;
    mode = 0;
    size = 100;
    deg = 32;

    small_size = 10;
    large_size = 100;

    datatype = __FLOAT__;
    opt_mode = GENERIC;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline long mid_num(const std::string& s) {
    return std::strtol(&s[s.find('_') + 1], nullptr, 10);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t get_size_in_bytes(const std::string& s)
{
    size_t mult = 1;
    if (s.find("KB") != std::string::npos) {
        mult = 1024;
    }
    if (s.find("MB") != std::string::npos) {
        mult = 1024*1024;
    }
    if (s.find("GB") != std::string::npos) {
        mult = 1024*1024*1024;
    }
    int short_num = mid_num(s);
    return mult * short_num;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::parse_args(int _argc, char **_argv)
{
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);

        if ((option.compare("-r") == 0) || (option.compare("-radius") == 0))
        {
            radius = atoi(_argv[++i]);
        }

        if ((option.compare("-m") == 0) || (option.compare("-mode") == 0))
        {
            mode = atoi(_argv[++i]);
        }

        if ((option.compare("-d") == 0) || (option.compare("-deg") == 0))
        {
            deg = atoi(_argv[++i]);
        }

        if ((option.compare("-s") == 0) || (option.compare("-size") == 0))
        {
            size = atoi(_argv[++i]);
        }

        if ((option.compare("-opt-mode") == 0))
        {
            string opt_str = _argv[++i];
            if(opt_str == "opt" || opt_str == "optimized")
                opt_mode = OPTIMIZED;

            if(opt_str == "gen" || opt_str == "generic")
                opt_mode = GENERIC;
        }

        if ((option.compare("-dt") == 0) || (option.compare("-datatype") == 0))
        {
            string dt = _argv[++i];

            if(dt == "float" || dt == "flt")
                datatype = __FLOAT__;

            if(dt == "double" || dt == "dbl")
                datatype = __DOUBLE__;
        }

        if (option.compare("-small-size") == 0)
        {
            small_size = get_size_in_bytes(_argv[++i]);
        }

        if (option.compare("-large-size") == 0)
        {
            large_size = get_size_in_bytes(_argv[++i]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
