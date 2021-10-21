#include "common/lib.h"

typedef double base_type;

struct CSRMatrix
{
    base_type *vals;
    int *row_ptrs;
    int *col_ids;
    int size;
    int elements;
};



#include "spmv.h"

void init(CSRMatrix &_matrix, int _size, int _deg)
{
    int elems = 0;
    vector<int> row_sizes(_size);
    for(int i = 0; i < _size; i++)
    {
        row_sizes[i] = _deg;//rand() % (2*_deg);
        elems += row_sizes[i];
    }

    _matrix.size = _size;
    _matrix.elements = elems;

    MemoryAPI::allocate_array(&_matrix.row_ptrs, (_matrix.size + 1));
    MemoryAPI::allocate_array(&_matrix.vals, _matrix.elements);
    MemoryAPI::allocate_array(&_matrix.col_ids, _matrix.elements);

    int ptr = 0;
    _matrix.row_ptrs[0] = 0;
    for(int i = 0; i < _size; i++)
    {
        for(int j = 0; j < row_sizes[i]; j++)
        {
            base_type r = rand() % 6 - 3;
            _matrix.vals[ptr] = r;
            _matrix.col_ids[ptr] = rand() % _size;
            ptr++;
        }
        _matrix.row_ptrs[i + 1] = ptr;
    }
}

void print(CSRMatrix &_matrix)
{
    int size = _matrix.size;
    base_type *vals = _matrix.vals;
    int *row_ptrs = _matrix.row_ptrs;
    int *col_ids = _matrix.col_ids;

    for(int i = 0; i < _matrix.size + 1; i++)
        cout << _matrix.row_ptrs[i] << " ";
    cout << endl;

    for(int i = 0; i < size; i++)
    {
        for(int j = row_ptrs[i]; j < row_ptrs[i+1]; ++j)
        {
            if(col_ids[j] < 0 || col_ids[j] >= size)
                cout << "(" << vals[j] << ", " << col_ids[j] << ") ";
        }
        cout << endl;
    }
}

void free(CSRMatrix &_matrix)
{
    MemoryAPI::free_array(_matrix.row_ptrs);
    MemoryAPI::free_array(_matrix.vals);
    MemoryAPI::free_array(_matrix.col_ids);
}

void call_kernel(ParserBenchmark &parser)
{
    size_t size = parser.get_size();
    size_t deg = parser.get_deg();

    CSRMatrix matrix;

    #ifdef METRIC_RUN
    int iterations = LOC_REPEAT * USUAL_METRICS_REPEAT;
    #else
    int iterations = LOC_REPEAT;
    #endif

    init(matrix, size, deg);

    base_type *x, *y;
    MemoryAPI::allocate_array(&x, matrix.size);
    MemoryAPI::allocate_array(&y, matrix.size);

    for(int i = 0; i < matrix.size; i++)
    {
        x[i] = rand() % 6 - 3;
        y[i] = rand() % 6 - 3;
    }

    #ifndef METRIC_RUN
    size_t flops_requested = matrix.elements * 2;
    size_t bytes_requested = flops_requested * 6;
    auto counter = PerformanceCounter(bytes_requested, flops_requested);
    #endif

	for(int i = 0; i < iterations; i++)
	{
        #ifndef METRIC_RUN
		//locality::utils::CacheAnnil(0);

        counter.start_timing();
        #endif

        kernel(matrix, x, y, parser.get_mode());

        #ifndef METRIC_RUN
        counter.end_timing();

        counter.update_counters();
        counter.print_local_counters();
        #endif
	}

    #ifndef METRIC_RUN
    counter.print_average_counters(true);
    #endif

    free(matrix);

    MemoryAPI::free_array(x);
    MemoryAPI::free_array(y);
}

int main(int argc, char **argv)
{
    ParserBenchmark parser;
    parser.parse_args(argc, argv);

    call_kernel(parser);
    return 0;
}

