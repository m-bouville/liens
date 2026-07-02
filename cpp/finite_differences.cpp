// #include <vector>

#include "finite_differences.hpp"

#include <cmath>

#include "field.hpp"

// using std::vector;



// derivatives (gradient, Laplacian)    
// void FD::gradient_x(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());

//     for (int j = 0; j < in.ny(); ++j)
//     {
//         for (int i = 0; i < in.nx(); ++i)
//         {
//             int ip = (i + 1) % in.nx();
//             int im = (i - 1 + in.nx()) % in.nx();

//             out[in.index(i, j)] =
//                 0.5 * (in(ip, j) - in(im, j));
//         }
//     }
// }

// void FD::gradient_y(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());

//     for (int j = 0; j < in.ny(); ++j)
//     {
//         int jp = (j + 1) % in.ny();
//         int jm = (j - 1 + in.ny()) % in.ny();

//         for (int i = 0; i < in.nx(); ++i)
//         {
//             out[in.index(i, j)] =
//                 0.5 * (in(i, jp) - in(i, jm));
//         }
//     }
// }

double FD::gradient_sqr(const Field& in)
{
    double out = 0;

    for (int j = 0; j < in.ny(); ++j)
        {
            int jp = (j + 1) % in.ny();
            int jm = (j - 1 + in.ny()) % in.ny();

            for (int i = 0; i < in.nx(); ++i)
            {
                int ip = (i + 1) % in.nx();
                int im = (i - 1 + in.nx()) % in.nx();

                double gx = 0.5 * (in(ip, j) - in(im, j));
                double gy = 0.5 * (in(i, jp) - in(i, jm));

                out += gx * gx + gy * gy;
            }
        }
    return out;
}

// void FD::gradient_sqr(const Field& in, Field& out)
// {
//     assert(in.nx() == out.nx());
//     assert(in.ny() == out.ny());

//     for (int j = 0; j < in.ny(); ++j)
//         {
//             int jp = (j + 1) % in.ny();
//             int jm = (j - 1 + in.ny()) % in.ny();

//             for (int i = 0; i < in.nx(); ++i)
//             {
//                 int ip = (i + 1) % in.nx();
//                 int im = (i - 1 + in.nx()) % in.nx();

//                 double gx = 0.5 * (in(ip, j) - in(im, j));
//                 double gy = 0.5 * (in(i, jp) - in(i, jm));

//                 out(i, j) = gx * gx + gy * gy;
//             }
//         }
// }

void FD::laplacian(const Field& in, Field& out)
{   
    assert(in.nx() == out.nx());
    assert(in.ny() == out.ny());

    for (int j = 0; j < in.ny(); ++j)
    {
        int jp = (j + 1) % in.ny();
        int jm = (j - 1 + in.ny()) % in.ny();

        for (int i = 0; i < in.nx(); ++i)
        {
            int ip = (i + 1) % in.nx();
            int im = (i - 1 + in.nx()) % in.nx();

            out(i, j) =
                in(ip, j) +
                in(im, j) +
                in(i, jp) +
                in(i, jm) -
                4.0 * in(i, j);
        }
    }
}



double FD::avg_gradient(const Field& in)
{
    double sum_grad = 0;

    for (int j = 0; j < in.ny(); ++j)
        {
            int jp = (j + 1) % in.ny();
            int jm = (j - 1 + in.ny()) % in.ny();

            for (int i = 0; i < in.nx(); ++i)
            {
                int ip = (i + 1) % in.nx();
                int im = (i - 1 + in.nx()) % in.nx();

                double gx = 0.5 * (in(ip, j) - in(im, j));
                double gy = 0.5 * (in(i, jp) - in(i, jm));

                sum_grad += std::sqrt(gx * gx + gy * gy);
            }
        }
    return (sum_grad / in.size());
}


void FD::structureTensor(const Field& phi,
                               Field& Jxx,
                               Field& Jxy,
                               Field& Jyy);
// TODO (for statistics)


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