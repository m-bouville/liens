#pragma once

#include <vector>

#include "field.hpp"




// derivatives (gradient, Laplacian)
namespace FD
{
    // Calculate periodic boundaries once and for all
    class Neighbors
    {
    public:
        std::vector<int> ip, im;
        std::vector<int> jp, jm;

        Neighbors(int nx, int ny)
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
    };


    // void gradient_x  (const Field& in, Field& out);
    // void gradient_y  (const Field& in, Field& out);
    double gradient_sqr(const Field& phi, const Neighbors& nb);
    // void gradient_sqr(const Field& in, Field& out);
    void laplacian   (const Field& in, Field& out, const Neighbors& nb);


    // for statistics
    struct StructureTensor
    {
        double Jxx_avg;
        double Jxy_avg;
        double Jyy_avg;

        double lambda1;      // largest eigenvalue
        double lambda2;      // smallest eigenvalue

        double anisotropy;   // (λ1-λ2)/(λ1+λ2)
        double angle;        // radians in [-π/2, π/2]
    };

    StructureTensor structure_tensor(const Field& phi, const Neighbors& nb);

    
    // for statistics
    double avg_gradient(const Field& phi, const Neighbors& nb);

    // characteristic length
    std::vector<double> autocorrelation(const Field& phi, int max_dist);

    // void chemical_potential(const Field& phi,
    //                         Field&       mu,
    //                         double       kappa,
    //                         LocalDerivative dfdphi);

    
    // Memory management: avoid repeated reallocations
    struct Workspace
    {
        Workspace(int nx, int ny)
            : df_dphi    (nx, ny),
            laplacian_phi(nx, ny),
            mu           (nx, ny),
            laplacian_mu (nx, ny),
            gradient_sqr (nx, ny)
        {}

        Field df_dphi;      // d f  / d phi
        Field laplacian_phi;// laplacian of phi

        Field mu;           // chemical potential = derivative (potential + gradient)
        Field laplacian_mu; // laplacian of chemical potential
        
        Field gradient_sqr; // for energy
    };


}