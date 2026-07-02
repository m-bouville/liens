#pragma once

#include <vector>

#include "field.hpp"




// derivatives (gradient, Laplacian)
namespace FD
{
    // void gradient_x  (const Field& in, Field& out);
    // void gradient_y  (const Field& in, Field& out);
    double gradient_sqr(const Field& in);
    // void gradient_sqr(const Field& in, Field& out);
    void laplacian   (const Field& in, Field& out);


    // TODO (for statistics):
    void structureTensor(const Field& phi,
                        Field& Jxx,
                        Field& Jxy,
                        Field& Jyy);

    
    // for statistics
    double avg_gradient(const Field& in);


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