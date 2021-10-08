#pragma once

void csr_spmv(CSRMatrix &_matrix, base_type *_x, base_type *_y)
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

void kernel(CSRMatrix &_matrix, base_type *_x, base_type *_y)
{
    csr_spmv(_matrix, _x, _y);
}
