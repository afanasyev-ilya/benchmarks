#pragma once

template <typename _T>
inline void csr_spmv(CSRMatrix &_matrix, _T *_x, _T *_y)
{
    int size = _matrix.size;
    base_type *vals = _matrix.vals;
    int *row_ptrs = _matrix.row_ptrs;
    int *col_ids = _matrix.col_ids;

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < size; i++)
    {
        for(int j = row_ptrs[i]; j < row_ptrs[i+1]; ++j)
        {
            _y[i] += vals[j] * _x[col_ids[j]];
        }
    }
}

template <typename _T>
inline void csr_spmv_unrollled(CSRMatrix &_matrix, _T *_x, _T *_y)
{
    int size = _matrix.size;
    base_type *vals = _matrix.vals;
    int *row_ptrs = _matrix.row_ptrs;
    int *col_ids = _matrix.col_ids;

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < size; i++)
    {
        for(int j = row_ptrs[i]; j < (row_ptrs[i+1] - 8); j += 8)
        {
            _y[i] += vals[j + 0] * _x[col_ids[j + 0]];
            _y[i] += vals[j + 1] * _x[col_ids[j + 1]];
            _y[i] += vals[j + 2] * _x[col_ids[j + 2]];
            _y[i] += vals[j + 3] * _x[col_ids[j + 3]];
            _y[i] += vals[j + 4] * _x[col_ids[j + 4]];
            _y[i] += vals[j + 5] * _x[col_ids[j + 5]];
            _y[i] += vals[j + 6] * _x[col_ids[j + 6]];
            _y[i] += vals[j + 7] * _x[col_ids[j + 7]];
        }

        for(int j = max(row_ptrs[i+1] - 8, row_ptrs[i]); j < row_ptrs[i+1]; j += 8)
        {
            _y[i] = vals[j] * _x[col_ids[j]] + _y[i];
        }
    }
}

template <typename _T>
inline void kernel(CSRMatrix &_matrix, _T *_x, _T *_y, int _mode)
{
    if(mode == 0)
        csr_spmv(_matrix, _x, _y);
    else if(mode == 1)
        csr_spmv_unrolled(_matrix, _x, _y);
}
