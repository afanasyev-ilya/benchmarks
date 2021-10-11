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
