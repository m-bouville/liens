#include "statistics.hpp"

#include <iostream>

#include <vector>
#include <cmath>

#include <limits>
#include <fftw3.h>

#include "field.hpp"
#include "finite_differences.hpp"
// #include "potential.hpp"
// #include "solver.hpp"
// #include "snapshot_writer.hpp"   // writer namespace



// Anisotropy
statistics::StructureTensor statistics::structure_tensor(const Field& phi, const FD::Neighbors& nb)
{
    statistics::StructureTensor out{};

    const int nx = phi.nx();
    const int ny = phi.ny();

    for (int j = 0; j < ny; ++j)
    {
        const int jp = nb.jp[j];
        const int jm = nb.jm[j];

        for (int i = 0; i < nx; ++i)
        {
            const int ip = nb.ip[i];
            const int im = nb.im[i];

            const double gx = 0.5 * (phi(ip, j) - phi(im, j));
            const double gy = 0.5 * (phi(i, jp) - phi(i, jm));

            out.Jxx_avg += gx * gx;
            out.Jxy_avg += gx * gy;
            out.Jyy_avg += gy * gy;
        }
    }

    const double invN = 1.0 / (nx * ny);

    out.Jxx_avg *= invN;
    out.Jxy_avg *= invN;
    out.Jyy_avg *= invN;

    out.trace = out.Jxx_avg + out.Jyy_avg;
    const double det = out.Jxx_avg * out.Jyy_avg - out.Jxy_avg * out.Jxy_avg;

    const double disc = std::sqrt(std::max(0.0, out.trace * out.trace - 4.0 * det));

    out.lambda1 = 0.5 * (out.trace + disc);
    out.lambda2 = 0.5 * (out.trace - disc);

    if (out.trace > 1e-12)
        out.anisotropy = (out.lambda1 - out.lambda2) / out.trace;
    else
        out.anisotropy = 0.0;

    out.angle = 0.5 * std::atan2(2.0 * out.Jxy_avg,
                                 out.Jxx_avg - out.Jyy_avg);

    return out;
}


// characteristic length
//    based on FFT
statistics::AutoCorrelMetrics statistics::autocorrelation(const Field& phi)
{
    const int Nx = phi.nx();
    const int Ny = phi.ny();
    const int N  = Nx * Ny;

    //------------------------------------------
    // subtract mean
    //------------------------------------------

    double mean = 0.0;
    for (double v : phi.values())
        mean += v;
    mean /= N;

    double* in =
        (double*) fftw_malloc(sizeof(double) * N);

    fftw_complex* freq =
        (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));

    double* corr =
        (double*) fftw_malloc(sizeof(double) * N);

    for (int i = 0; i < N; ++i)
        in[i] = phi.values()[i] - mean;

    //------------------------------------------
    // FFT
    //------------------------------------------

    fftw_plan forward =
        fftw_plan_dft_r2c_2d(
            Ny, Nx,
            in,
            freq,
            FFTW_ESTIMATE);

    fftw_execute(forward);

    //------------------------------------------
    // Power spectrum
    //------------------------------------------

    const int Nk = Ny * (Nx/2 + 1);

    for (int i = 0; i < Nk; ++i)
    {
        double re = freq[i][0];
        double im = freq[i][1];

        freq[i][0] = re*re + im*im;
        freq[i][1] = 0.0;
    }

    //------------------------------------------
    // inverse FFT
    //------------------------------------------

    fftw_plan backward =
        fftw_plan_dft_c2r_2d(
            Ny, Nx,
            freq,
            corr,
            FFTW_ESTIMATE);

    fftw_execute(backward);

    // std::cout << corr[0] << " (before normalizing)\n";

    // normalize
    const double invN2 = 1.0 / (double(N) * double(N));
    for (int i = 0; i < N; ++i)
        corr[i] *= invN2;

    
    double var = 0.0;
    for (double v : phi.values())
    {
        double d = v - mean;
        var += d*d;
    }
    var /= N;


    //------------------------------------------
    // radial average
    //------------------------------------------

    int max_dist = std::min(Nx*2/3, Ny*2/3);

    std::vector<double> radial(max_dist + 1, 0.0);
    std::vector<int>    counts(max_dist + 1, 0);

    auto periodic = [](int x, int n)
    {
        if (x > n/2)
            return x - n;
        return x;
    };

    for (int y = 0; y < Ny; ++y)
    {
        int yy = periodic(y, Ny);

        for (int x = 0; x < Nx; ++x)
        {
            int xx = periodic(x, Nx);

            double r  = std::sqrt(double(xx*xx + yy*yy));
            int    ir = int(std::round(r));

            if (ir <= max_dist)
            {
                radial[ir] += corr[y*Nx + x];
                counts[ir]++;
            }
        }
    }

    for (int r = 0; r <= max_dist; ++r)
        if (counts[r] > 0)
            radial[r] /= counts[r];


    // normalize by C(0)
    // std::cout <<  "variance = " << var << ", corr[0]: " << corr[0] 
    //           << ", radial[0]: " << radial[0] << '\n';
    if (std::abs(radial[0]) > 1e-15)
    {
        for (double& v : radial)
            v /= radial[0];
    }

    //------------------------------------------
    // locate first peak
    //------------------------------------------

    AutoCorrelMetrics out;
    out.values = radial;
    out.peak_distance = -1;  // even not overwritten: failure

    bool seen_negative = false;

    for (int r = 3; r < max_dist-3; ++r)
    {
        if (radial[r] < 0.0)
        {
            seen_negative = true;
            continue;
        }

        if (!seen_negative)
            continue;

        double left  = (radial[r-3] + radial[r-2] + radial[r-1]) / 3.0;
        double center= (radial[r-1] + radial[r  ] + radial[r+1]) / 3.0;
        double right = (radial[r+1] + radial[r+2] + radial[r+3]) / 3.0;

        if (center > 1.1)  // should be <= 1
        {
            std::cout << "Warning: r:" << r << ", left:" << left
                      << ", center:" << center << ", right:" << right << '\n';
        }

        if (center > left &&
            center > right)
        {
            out.peak_distance = r;
            out.peak_value    = center;   // smoother than radial[r];
            break;
        }
    }
    
    if (out.peak_distance == -1)  // nothing found
    {
        out.peak_distance= max_dist;
        out.peak_value   = 0.;  // /!\ arbitrary
    }

    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);

    fftw_free(in);
    fftw_free(freq);
    fftw_free(corr);

    return out;
}
