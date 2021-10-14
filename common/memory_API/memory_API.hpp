#ifdef __USE_INTEL__
#include <mm_malloc.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::allocate_array(_T **_ptr, size_t _size)
{
    #ifdef __USE_INTEL__
    *_ptr = (_T*)_mm_malloc(sizeof(_T)*(_size),2097152);
    #else
    *_ptr = new _T[_size];
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::allocate_host_array(_T **_ptr, size_t _size)
{
    *_ptr = (_T*)malloc(_size*sizeof(_T));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::free_array(_T *_ptr)
{
    delete []_ptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::free_host_array(_T *_ptr)
{
    if(_ptr != NULL)
    {
        free(_ptr);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::copy(_T *_dst, _T *_src, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long i = 0; i < _size; i++)
    {
        _dst[i] = _src[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::set(_T *_data, _T _val, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < _size; i++)
    {
        _data[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void MemoryAPI::resize(_T **_ptr, size_t _new_size)
{
    if(*_ptr != NULL)
        MemoryAPI::free_array(*_ptr);
    MemoryAPI::allocate_array(_ptr, _new_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::move_array_to_device(_T *_ptr, size_t _size)
{
    int device_id = 0;
    SAFE_CALL(cudaGetDevice(&device_id));
    SAFE_CALL(cudaMemPrefetchAsync(_ptr, _size*sizeof(_T), device_id, NULL));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void MemoryAPI::move_array_to_host(_T *_ptr, size_t _size)
{
    SAFE_CALL(cudaMemPrefetchAsync(_ptr, _size*sizeof(_T), cudaCpuDeviceId, NULL));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
