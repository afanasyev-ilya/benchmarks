#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_size(string name, size_t len)
{
    string sizes[] = { "B", "KB", "MB", "GB", "TB" };
    int order = 0;
    while (len >= 1024 && order < 4) {
        order++;
        len = len/1024;
    }
    cout << name << " : " << len << " " << sizes[order] << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_omp_info()
{
    cout << "number of OMP threads used: " << omp_get_max_threads() << endl;
    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int cpu_num = sched_getcpu();
        printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERIC_LOAD(op_reg, op_offset, op_data) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    op_reg[] = op_data[(op_offset) + i];


#define GENERIC_LOAD_SH(op_reg, op_data) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    op_reg[i] = op_data[i];


#define GENERIC_COPY(dst, src) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    dst[i] = src[i];

#define GENERIC_STORE(reg, offset, data) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    data[offset + i] = reg[i];


#define GENERIC_SET_ZERO(op_reg) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    op_reg[i] = 0;

#define GENERIC_ADD_PS(op_res, op_arg1, op_arg2) \
_Pragma("simd")                                  \
_Pragma("vector")                                \
_Pragma("ivdep")                                 \
for(int i = 0; i < 16; i++)                      \
    op_res[i] = op_arg1[i] + op_arg2[i];
