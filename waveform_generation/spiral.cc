#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "spiral.h"


// Helper: linspace
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    if (num == 1) {
        result[0] = start;
        return result;
    }
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + step * i;
    }
    return result;
}

// Helper: diff
template <typename T>
std::vector<T> diff(const std::vector<T>& arr) {
    std::vector<T> result;
    if (arr.size() < 2) return result;
    result.reserve(arr.size() - 1);
    for (size_t i = 1; i < arr.size(); i++) {
        result.push_back(arr[i] - arr[i - 1]);
    }
    return result;
}

// Helper: pad with one zero at end
template <typename T>
std::vector<T> pad_end(const std::vector<T>& arr) {
    std::vector<T> result = arr;
    result.push_back(T(0));
    return result;
}

// Convert complex gradient -> array of (real, imag)
std::vector<std::vector<double>> traj_complex_to_array(const std::vector<std::complex<double>>& traj) {
    std::vector<std::vector<double>> result(traj.size(), std::vector<double>(2));
    for (size_t i = 0; i < traj.size(); i++) {
        result[i][0] = traj[i].real();
        result[i][1] = traj[i].imag();
    }
    return result;
}

// Return only gradient waveform
std::vector<std::vector<double>> spiral_gradient(double fov, double res, double gts, double gslew, double gamp,double gam) {


   /*Analytic Archimedean spiral designer. Produces trajectory, gradients,
    and slew rate. Gradient returned has units mT/m.

    Args:
        fov (float): imaging field of view in m.
        res (float): resolution, in m.
        gts (float): sample time in s.
        gslew (float): max slew rate in mT/m/ms.
        gamp (float): max gradient amplitude in mT/m.
        gam (float): gamma of current nucleus in Hz/mT

    References:
        Glover, G. H.(1999).
        Simple Analytic Spiral K-Space Algorithm.
        Magnetic resonance in medicine, 42, 412-415.

        Bernstein, M.A.; King, K.F.; amd Zhou, X.J. (2004).
        Handbook of MRI Pulse Sequences. Elsevier.
    */


    // gam = 267.522e6 / 1000.0;  // rad/s/mT
    double gambar = gam / (2.0 * M_PI);  // Hz/mT
    int N = static_cast<int>(fov / res);
    double lam = 1.0 / (2.0 * M_PI * fov);
    double beta = gambar * gslew / lam;

    double kmax = N / (2.0 * fov);
    double a_2 = pow(9.0 * beta / 4.0, 1.0 / 3.0);
    double lamb = 5.0;
    double theta_max = kmax / lam;
    double ts = pow((3.0 * gam * gamp) / (4.0 * M_PI * lam * pow(a_2, 2)), 3);
    double theta_s = 0.5 * beta * pow(ts, 2);
    theta_s /= (lamb + beta / (2.0 * a_2) * pow(ts, 4.0 / 3.0));
    double t_g = M_PI * lam * (pow(theta_max, 2) - pow(theta_s, 2)) / (gam * gamp);
    int n_s = static_cast<int>(std::round(ts / gts));
    int n_g = static_cast<int>(std::round(t_g / gts));

    std::vector<std::complex<double>> k;

    if (theta_max > theta_s) {
        std::cout << "Spiral trajectory is slewrate limited or amplitude limited\n";

        double tacq = ts + t_g;
        auto t_s = linspace(0, ts, n_s);
        auto t_gs = linspace(ts + gts, tacq, n_g);

        std::vector<double> theta_1(t_s.size());
        for (size_t i = 0; i < t_s.size(); i++) {
            theta_1[i] = (beta / 2.0) * pow(t_s[i], 2);
            theta_1[i] /= (lamb + beta / (2.0 * a_2) * pow(t_s[i], 4.0 / 3.0));
        }

        std::vector<double> theta_2(t_gs.size());
        for (size_t i = 0; i < t_gs.size(); i++) {
            theta_2[i] = sqrt(theta_s * theta_s + gam / (M_PI * lam) * gamp * (t_gs[i] - ts));
        }

        std::vector<std::complex<double>> k1(theta_1.size());
        for (size_t i = 0; i < theta_1.size(); i++) {
            k1[i] = lam * theta_1[i] * std::complex<double>(cos(theta_1[i]), sin(theta_1[i]));
        }

        std::vector<std::complex<double>> k2(theta_2.size());
        for (size_t i = 0; i < theta_2.size(); i++) {
            k2[i] = lam * theta_2[i] * std::complex<double>(cos(theta_2[i]), sin(theta_2[i]));
        }

        k.reserve(k1.size() + k2.size());
        k.insert(k.end(), k1.begin(), k1.end());
        k.insert(k.end(), k2.begin(), k2.end());

    } else {
        double tacq = 2 * M_PI * fov / 3 * sqrt(M_PI / (gam * gslew * pow(res, 3)));
        int n_t = static_cast<int>(std::round(tacq / gts));
        auto t_s = linspace(0, tacq, n_t);

        std::vector<double> theta_1(t_s.size());
        for (size_t i = 0; i < t_s.size(); i++) {
            theta_1[i] = (beta / 2.0) * pow(t_s[i], 2);
            theta_1[i] /= (lamb + beta / (2.0 * a_2) * pow(t_s[i], 4.0 / 3.0));
        }

        k.resize(theta_1.size());
        for (size_t i = 0; i < theta_1.size(); i++) {
            k[i] = lam * theta_1[i] * std::complex<double>(cos(theta_1[i]), sin(theta_1[i]));
        }
    }

    // Gradient only
    auto g_complex = pad_end(diff(k));
    for (auto& val : g_complex) {
        val /= (gts * gambar);
    }

    return traj_complex_to_array(g_complex);
}