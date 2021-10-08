#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    radius = 1;
    length = 100;
    mode = 0;
    size = 10;
    deg = 32;
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

        if ((option.compare("-l") == 0) || (option.compare("-length") == 0))
        {
            length = atoi(_argv[++i]);
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
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
