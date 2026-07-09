#ifndef __ISI_helperSpiral__
#define __ISI_helperSpiral__
#include <vector>
#include <complex> // ← Required for std::complex
#include <cmath>
std::vector<double> linspace(double start, double end, int num);
template <typename T>
std::vector<T> diff(const std::vector<T> &arr);
template <typename T>
std::vector<T> pad_end(const std::vector<T> &arr);
std::vector<std::vector<double>> traj_complex_to_array(const std::vector<std::complex<double>> &traj);

std::vector<std::vector<double>> spiral_gradient(double fov, double res, double gts, double gslew, double gamp, double gam);

#endif