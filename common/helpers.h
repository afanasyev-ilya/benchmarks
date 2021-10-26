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
op_reg[0] = op_data[(op_offset) + 0];  \
op_reg[1] = op_data[(op_offset) + 1];  \
op_reg[2] = op_data[(op_offset) + 2];  \
op_reg[3] = op_data[(op_offset) + 3];  \
op_reg[4] = op_data[(op_offset) + 4];  \
op_reg[5] = op_data[(op_offset) + 5];  \
op_reg[6] = op_data[(op_offset) + 6];  \
op_reg[7] = op_data[(op_offset) + 7];  \
op_reg[8] = op_data[(op_offset) + 8];  \
op_reg[9] = op_data[(op_offset) + 9];  \
op_reg[10] = op_data[(op_offset) + 10];  \
op_reg[11] = op_data[(op_offset) + 11];  \
op_reg[12] = op_data[(op_offset) + 12];  \
op_reg[13] = op_data[(op_offset) + 13];  \
op_reg[14] = op_data[(op_offset) + 14];  \
op_reg[15] = op_data[(op_offset) + 15];

#define GENERIC_COPY(dst, src) \
dst[0] = src[0];  \
dst[1] = src[1];  \
dst[2] = src[2];  \
dst[3] = src[3];  \
dst[4] = src[4];  \
dst[5] = src[5];  \
dst[6] = src[6];  \
dst[7] = src[7];  \
dst[8] = src[8];  \
dst[9] = src[9];  \
dst[10] = src[10];  \
dst[11] = src[11];  \
dst[12] = src[12];  \
dst[13] = src[13];  \
dst[14] = src[14];  \
dst[15] = src[15];

#define GENERIC_STORE(reg, offset, data) \
data[offset + 0] = reg[0];  \
data[offset + 1] = reg[1];  \
data[offset + 2] = reg[2];  \
data[offset + 3] = reg[3];  \
data[offset + 4] = reg[4];  \
data[offset + 5] = reg[5];  \
data[offset + 6] = reg[6];  \
data[offset + 7] = reg[7];  \
data[offset + 8] = reg[8];  \
data[offset + 9] = reg[9];  \
data[offset + 10] = reg[10];  \
data[offset + 11] = reg[11];  \
data[offset + 12] = reg[12];  \
data[offset + 13] = reg[13];  \
data[offset + 14] = reg[14];  \
data[offset + 15] = reg[15];


#define GENERIC_SET_ZERO(reg) \
reg[0] = 0;\
reg[1] = 0;\
reg[2] = 0;\
reg[3] = 0;\
reg[4] = 0;\
reg[5] = 0;\
reg[6] = 0;\
reg[7] = 0;\
reg[8] = 0;\
reg[9] = 0;\
reg[10] = 0;\
reg[11] = 0;\
reg[12] = 0;\
reg[13] = 0;\
reg[14] = 0;\
reg[15] = 0;

#define GENERIC_ADD_PS(op_res, op_arg1, op_arg2) \
op_res[0] = op_arg1[0] + op_arg2[0];\
op_res[1] = op_arg1[1] + op_arg2[1];\
op_res[2] = op_arg1[2] + op_arg2[2];\
op_res[3] = op_arg1[3] + op_arg2[3];\
op_res[4] = op_arg1[4] + op_arg2[4];\
op_res[5] = op_arg1[5] + op_arg2[5];\
op_res[6] = op_arg1[6] + op_arg2[6];\
op_res[7] = op_arg1[7] + op_arg2[7];\
op_res[8] = op_arg1[8] + op_arg2[8];\
op_res[9] = op_arg1[9] + op_arg2[9];\
op_res[10] = op_arg1[10] + op_arg2[10];\
op_res[11] = op_arg1[11] + op_arg2[11];\
op_res[12] = op_arg1[12] + op_arg2[12];\
op_res[13] = op_arg1[13] + op_arg2[13];\
op_res[14] = op_arg1[14] + op_arg2[14];\
op_res[15] = op_arg1[15] + op_arg2[15];
