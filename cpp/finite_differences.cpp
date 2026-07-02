#include "finite_differences.hpp"

#include <vector>
#include <cmath>
#include <limits>

#include "field.hpp"



FD::Neighbors::Neighbors(int nx, int ny)
            : ip(nx), im(nx), jp(ny), jm(ny)
{
    for (int i = 0; i < nx; ++i)
    {
        ip[i] = (i + 1) % nx;
        im[i] = (i - 1 + nx) % nx;
    }

    for (int j = 0; j < ny; ++j)
    {
        jp[j] = (j + 1) % ny;
        jm[j] = (j - 1 + ny) % ny;
    }
}

double FD::gradient_sqr(const Field& phi, const Neighbors& nb)
{
    double out = 0;

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

            double gx = 0.5 * (phi(ip, j) - phi(im, j));
            double gy = 0.5 * (phi(i, jp) - phi(i, jm));

            out += gx * gx + gy * gy;
        }
    }
    return out;
}


void FD::laplacian(const Field& in, Field& out, const Neighbors& nb)
{   
    assert(in.nx() == out.nx());
    assert(in.ny() == out.ny());

    const int nx = in.nx();
    const int ny = in.ny();

    for (int j = 0; j < ny; ++j)
    {
        const int jp = nb.jp[j];
        const int jm = nb.jm[j];

        for (int i = 0; i < nx; ++i)
        {
            const int ip = nb.ip[i];
            const int im = nb.im[i];

            out(i, j) =
                in(ip, j) +
                in(im, j) +
                in(i, jp) +
                in(i, jm) -
                4.0 * in(i, j);
        }
    }
}



double FD::avg_gradient(const Field& phi, const Neighbors& nb)
{
    double sum_grad = 0;

    const int nx = phi.nx();
    const int ny = phi.ny();

    for (int j = 0; j < phi.ny(); ++j)
    {
        const int jp = nb.jp[j];
        const int jm = nb.jm[j];

        for (int i = 0; i < nx; ++i)
        {
            const int ip = nb.ip[i];
            const int im = nb.im[i];

            double gx = 0.5 * (phi(ip, j) - phi(im, j));
            double gy = 0.5 * (phi(i, jp) - phi(i, jm));

            sum_grad += std::sqrt(gx * gx + gy * gy);
        }
    }
    return (sum_grad / phi.size());
}


// for statistics (anisotropy)
FD::StructureTensor FD::structure_tensor(const Field& phi, const Neighbors& nb)
{
    FD::StructureTensor out{};

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
FD::AutoCorrelMetrics FD::autocorrelation(const Field& phi, int max_dist)
{
    const auto& f  = phi.values();   // assume flattened 2D
    const int   Nx = phi.nx();
    const int   Ny = phi.ny();
    const int   N  = Nx * Ny;

    max_dist = std::min(max_dist, std::max(Nx, Ny));
    const int   max_dist2 = max_dist*max_dist;

    auto idx = [&](int x, int y)
    {
        return (x + Nx) % Nx + ((y + Ny) % Ny) * Nx; // periodic
    };

    std::vector<double> C    (max_dist + 1, 0.0);
    std::vector<int>    count(max_dist + 1, 0);

    // subtract mean (VERY important)
    double mean = 0.0;
    for (double v : f) mean += v;
    mean /= N;

    std::vector<double> fp(N);
    for (int i = 0; i < N; ++i)
        fp[i] = f[i] - mean;

    // brute-force correlations
    for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x)
        {
            int i0 = idx(x, y);

            for (int dy = -max_dist; dy <= max_dist; ++dy)
                for (int dx = 0;     dx <= max_dist; ++dx)  // avoid double-counting
                {
                    int r2 = dx*dx + dy*dy;
                    if (r2 > max_dist2) continue;

                    int r  = static_cast<int>(std::sqrt(r2 + 1e-12));

                    int i1 = idx(x + dx, y + dy);  // accounts for periodic boundaries

                    C    [r] += fp[i0] * fp[i1];
                    count[r] += 1;
                }
        }

    // normalize radial bins
    for (int r = 0; r <= max_dist; ++r)
    {
        if (count[r] > 0)
            C[r] /= count[r];
    }

    // normalize by C(0)
    double C0 = C[0];
    if (std::abs(C0) > 1e-12)
    {
        for (double& v : C)
            v /= C0;
    }

    // find first non-trivial peak (skip r=0)
    int    peak_r  =   1;
    double peak_val= C[1];

    bool   seen_negative  = false;

    for (int r = 3; r < max_dist - 3; ++r)
    {
        if (C[r-1] <= 0 || C[r] <= 0 || C[r+1] <= 0)
        {
            seen_negative  = true;  // we reach the well of the first min
            continue;
        } else if (!seen_negative) // we have not reach the well of the first min
            continue;               //  so we cannot be at the first max

        double center = (C[r-1] + C[r  ] + C[r+1]) / 3.0;
        double left   = (C[r-3] + C[r-2] + C[r-1]) / 3.0;
        double right  = (C[r+1] + C[r+2] + C[r+3]) / 3.0;

        if (center > left && center > right)
        {
            peak_r   = r;
            peak_val = C[r];
            break;
        }
    }

    // constructing output
    AutoCorrelMetrics out;
    out.values        = C;
    out.peak_distance = peak_r > 1 ? peak_r   :   max_dist;
    out.peak_value    = peak_r > 1 ? peak_val : C[max_dist];

    return out;
}
