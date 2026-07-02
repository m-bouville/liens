// #include <vector>

#include "finite_differences.hpp"

#include <vector>
#include <cmath>
#include <limits>

#include "field.hpp"

// using std::vector;



// derivatives (gradient, Laplacian)   
// void FD::gradient_x(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());

//     const int nx = phi.nx();

//     for (int i = 0; i < nx; ++i)
//     {
//         int ip = (i + 1) % nx;
//         int im = (i - 1 + nx) % nx;

//         for (int j = 0; j < in.ny(); ++j)
//             out[in.index(i, j)] = 0.5 * (in(ip, j) - in(im, j));
//     }
// }
// void FD::gradient_y(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());
    
//     const int ny = in.ny();

//     for (int j = 0; j < ny; ++j)
//     {     
//         int jp = (j + 1) % ny;
//         int jm = (j - 1 + ny) % ny;

//         for (int i = 0; i < in.nx(); ++i)
//             out[in.index(i, j)] = 0.5 * (in(i, jp) - in(i, jm));
//     }
// }

double FD::gradient_sqr(const Field& phi, const Neighbors& nb)
{
    double out = 0;

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

            out += gx * gx + gy * gy;
        }
    }
    return out;
}

// void FD::gradient_sqr(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());

//     const int nx = in.nx();
//     const int ny = in.ny();

//     for (int j = 0; j < ny; ++j)
//     {
//         int jp = (j + 1) % ny;
//         int jm = (j - 1 + ny) % ny;

//         for (int i = 0; i < nx; ++i)
//         {
//             int ip = (i + 1) % nx;
//             int im = (i - 1 + nx) % nx;

//             double gx = 0.5 * (in(ip, j) - in(im, j));
//             double gy = 0.5 * (in(i, jp) - in(i, jm));

//             out(i, j) = gx * gx + gy * gy;
//         }
//     }
// }

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



// void FD::all_gradients(const Field& phi, 
//                              Field& grad_x,
//                              Field& grad_y,                            
//                              Field& grad_sqr,
//                              Field& laplacian)
// {   
//     assert(phi.nx() == grad_x.nx() == grad_y.nx() == grad_sqr.nx() == laplacian.nx());
//     assert(phi.ny() == grad_x.ny() == grad_y.ny() == grad_sqr.ny() == laplacian.ny());

//     const int nx = phi.nx();
//     const int ny = phi.ny();

//     for (int j = 0; j < ny; ++j)
//     {
//         int jp = (j + 1) % ny;
//         int jm = (j - 1 + ny) % ny;

//         for (int i = 0; i < nx; ++i)
//         {
//             int ip = (i + 1) % nx;
//             int im = (i - 1 + nx) % nx;

            
//             grad_x (i, j) = 0.5 * (phi(ip, j) - phi(im, j));
//             grad_y (i, j) = 0.5 * (phi(i, jp) - phi(i, jm));

//             grad_sqr (i, j) = grad_x(i, j) * grad_x(i, j) + 
//                               grad_y(i, j) * grad_y(i, j);

//             laplacian(i, j) =
//                 phi(ip, j) +
//                 phi(im, j) +
//                 phi(i, jp) +
//                 phi(i, jm) -
//                 4.0 * phi(i, j);
//         }
//     }
// }



// double FD::gradient_x(const Field& phi, int i, int j)
// {
//     int ip = (i + 1) % phi.nx();
//     int im = (i - 1 + phi.nx()) % phi.nx();

//     return 0.5 * (phi(ip, j) - phi(im, j));
// }
// double FD::gradient_y(const Field& phi, int i, int j)
// {
//     int jp = (j + 1) % phi.ny();
//     int jm = (j - 1 + phi.ny()) % phi.ny();

//     return 0.5 * (phi(i, jp) - phi(i, jm));
// }

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

    const double tr  = out.Jxx_avg + out.Jyy_avg;
    const double det = out.Jxx_avg * out.Jyy_avg - out.Jxy_avg * out.Jxy_avg;

    const double disc = std::sqrt(std::max(0.0, tr * tr - 4.0 * det));

    out.lambda1 = 0.5 * (tr + disc);
    out.lambda2 = 0.5 * (tr - disc);

    if (tr > 1e-12)
        out.anisotropy = (out.lambda1 - out.lambda2) / tr;
    else
        out.anisotropy = 0.0;

    out.angle = 0.5 * std::atan2(2.0 * out.Jxy_avg,
                                 out.Jxx_avg - out.Jyy_avg);

    return out;
}


// characteristic length
std::vector<double> FD::autocorrelation(const Field& phi, int max_dist)
{
    const auto& f  = phi.values();   // assume flattened 2D
    const int   Nx = phi.nx();
    const int   Ny = phi.ny();
    const int   N  = Nx * Ny;
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
    int    peak_r   = 1;
    double peak_val = C[1];

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
            peak_r = r;
            peak_val = C[r];
            break;
        }
    }

    // append peak info
    C.insert(C.begin(), (double)peak_val);
    C.insert(C.begin(), (double)peak_r);

    return C;
}






// template<class LocalDerivative>
// void FD::chemical_potential(const Field& phi,
//                         Field& mu,
//                         double kappa,
//                         LocalDerivative dfdphi)
// {
//     FD::laplacian(phi, lap);

//     for (int j = 0; j < phi.ny(); ++j)
//     {
//         int jp = (j + 1) % phi.ny();
//         int jm = (j - 1 + phi.ny()) % phi.ny();

//         for (int i = 0; i < phi.nx(); ++i)
//         {
//             int ip = (i + 1) % phi.nx();
//             int im = (i - 1 + phi.nx()) % phi.nx();

//             double lap =
//                 phi(ip,j) + phi(im,j) +
//                 phi(i,jp) + phi(i,jm)
//                 - 4.0 * phi(i,j);

//             mu(i,j) =
//                 dfdphi(phi(i,j))
//                 - kappa * lap;
//         }
//     }
// }