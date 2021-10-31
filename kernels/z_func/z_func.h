#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>
#ifdef __USE_INTEL__
#include <immintrin.h>
#endif

using std::string;

void init(char *str, size_t size)
{
    srand(time(0));
    for (size_t i = 0; i < size - 1; i++) {
        str[i] = 'a' + rand() % 26;
    }
    str[size - 1] = '\0';
}

pair<int, int> z_function_count (vector<double> &z, char *s, int n) {
    int flops = 0, bytes = 0;
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r) {
            z[i] = min((int)(r-i+1), (int)(z[i-l]));
            bytes += 2 * sizeof(char);
        }
        while (i+z[i] < n && s[(int)z[i]] == s[i+(int)z[i]]) {
            ++z[i];
            bytes += 2 * sizeof(char) + 2 * sizeof(double);
            flops += 1;
        }
        if (i+z[i]-1 > r) {
            bytes += 1 * sizeof(double);
            flops += 1;
            l = i, r = i + z[i] - 1;
        }
    }
    return {flops, bytes};
}

void z_function (vector<double> &z, char *s, int n) {
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r)
            z[i] = min((int)(r-i+1), (int)(z[i-l]));
        while (i+z[i] < n && s[(int)z[i]] == s[i+(int)z[i]])
            ++z[i];
        if (i+z[i]-1 > r)
            l = i, r = i+z[i]-1;
    }
}

void kernel(vector<double> &z, char *str, size_t size) {
    z_function(z, str, size);
}
