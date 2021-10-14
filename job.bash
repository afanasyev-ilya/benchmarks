#!/bin/bash

if [ -e ../vars_global.bash ]; then
    echo ../vars_global.bash exists
    source ../vars_global.bash
fi
if [ -e ./vars.bash ]; then
    echo ./vars.bash exists
    source ./vars.bash
fi

export PATH=/home/z44377r/arm_gcc/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/z44377r/arm_gcc/lib/:/home/z44377r/arm_gcc/lib64/

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

export PATH=/home/z44377r/ARM/gcc_latest/bin:$PATH
export INCLUDE=/home/z44377r/ARM/gcc_latest/include:$INCLUDE
export LD_LIBRARY_PATH=/home/z44377r/ARM/gcc_latest/lib64:$LD_LIBRARY_PATH

lscpu

make gather_ker ARCH=a64fx CXX=g++

./bin/gather_ker -small-size 8KB -large-size 1GB
#./bin/gather_ker -small-size 64KB -large-size 1GB
#./bin/gather_ker -small-size 256KB -large-size 1GB
#./bin/gather_ker -small-size 2MB -large-size 1GB

#python3 ./run_tests.py --bench=gather --arch=a64fx