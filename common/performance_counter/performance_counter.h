#ifndef TIMERS_H
#define TIMERS_H
#include <ios>
#include <chrono>

class PerformanceCounter {
    double total_time = 0;
    double total_bw = 0;
    double total_flops = 0;
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
    double local_bw = 0;
    double local_time = 0;
    double local_flops = 0;
    unsigned long long bytes_requested;
    unsigned long long flops_executed;
    double type = 0;
public:
    PerformanceCounter(unsigned long long bytes, unsigned long long flops):bytes_requested(bytes), flops_executed(flops) {
        type = flops_executed / bytes_requested;
    }
    void start_timing(void) {
        start_time = std::chrono::steady_clock::now();
    }

    void end_timing(void) {
        end_time = std::chrono::steady_clock::now();
    }

    void force_update_counters(double force_time, double force_bw, double force_flops)
    {
        local_time = force_time;
        local_bw = force_bw;
        local_flops = force_flops;
        total_bw += local_bw;
        total_time += local_time;
        total_flops += local_flops;
    }

    void update_counters() {
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        local_time = elapsed_seconds.count();
        local_bw = bytes_requested * 1e-9 / local_time;
        local_flops = flops_executed * 1e-9 / local_time;
        total_bw += local_bw;
        total_time += local_time;
        total_flops += local_flops;
    }

    void print_local_counters(void) {
        std::cout << std::fixed;
        std::cout << "local_time: " << local_time << " s" << std::endl;
        std::cout << "local_bw: " << local_bw << " Gb/s" << std::endl;
        std::cout << "local_flops: " << local_flops << " GFlops" << std::endl << std::endl;
    }

    void print_average_counters(bool flops_required) {
        std::cout << std::fixed;
        std::cout << "avg_time: " << total_time/LOC_REPEAT << " s" << std::endl;
        std::cout << "avg_bw: " << total_bw/LOC_REPEAT << " Gb/s" << std::endl;
        if (flops_required) {
            std::cout << "avg_flops: " << total_flops/LOC_REPEAT << " GFlops" << std::endl;
        }
        std::cout << "flops_per_byte: " << (double) flops_executed / (double) bytes_requested << std::endl;
    }

    void print_bw() {
        FILE *fres = fopen("bw.txt", "a");
        fprintf(fres, "%lf\n", total_bw/LOC_REPEAT);
        fclose(fres);
    }

    void print_flops()
    {
        FILE *fres = fopen("flops.txt", "a");
        fprintf(fres, "%lf\n", total_flops/LOC_REPEAT);
        fclose(fres);   
    }

    void print_time()
    {
        FILE *fres = fopen("time.txt", "a");
        fprintf(fres, "%lf\n", total_time/LOC_REPEAT);
        fclose(fres);   
    }
};

#endif
