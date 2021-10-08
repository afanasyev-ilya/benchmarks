#pragma once

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <vector>

#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#define LOC_REPEAT 8

#define GRAPH_METRICS_REPEAT 1000
#define USUAL_METRICS_REPEAT 20

using namespace std;

#include "timer/timer.h"
#include "cmd_parser/cmd_parser.h"
#include "performance_counter/performance_counter.h"
#include "memory_API/memory_API.h"