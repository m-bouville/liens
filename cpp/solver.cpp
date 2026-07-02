#include "solver.hpp"


// #include "field.hpp"
// #include "potential.hpp"


void FDSolver::step(     OrderParameter& op,
                   const Potential&      potential, 
                   const FD::Neighbors&  neighbors,
                         double          T,
                         double          dt)
    {
        int grid_size = op.size();

        // Local free-energy derivative
        potential.derivative(op, T, __ws.df_dphi);

        // Gradient contribution
       FD::laplacian(op.field(), __ws.laplacian_phi, neighbors);

        for (int i = 0; i < grid_size; ++i)
            __ws.mu[i] = __ws.df_dphi[i] - op.kappa() * __ws.laplacian_phi[i];


        // Evolution equation
        if (!op.is_conserved())  // Allen-Cahn
        {
            for (int i = 0; i < grid_size; ++i)
                op[i] -= dt * op.mobility() * __ws.mu[i];
        }
        else  // Cahn-Hilliard
        {
           FD::laplacian(__ws.mu, __ws.laplacian_mu, neighbors);

            for (int i = 0; i < grid_size; ++i)
                op[i] += dt * op.mobility() * __ws.laplacian_mu[i];
        }
    }


double FDSolver::energy(const OrderParameter& op,
                        const Potential&      potential, 
                        const FD::Neighbors&  neighbors,
                              double          T) const
    {
        return (potential.bulk_energy(op, T) + 
                0.5 * op.kappa() * FD::gradient_sqr(op.field(), neighbors));
    }
