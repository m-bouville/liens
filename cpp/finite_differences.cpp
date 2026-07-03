#include "finite_differences.hpp"

#include <vector>
#include <cmath>

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


