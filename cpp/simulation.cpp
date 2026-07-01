#include <iostream>

#include "simulation.hpp"

#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"

void Simulation::run()
{
    for (double T : __config.temperatures)
        for (double noise : __config.noises)
            for (int seed : __config.seeds)
            {
                std::cout << "T: " << T << ", noise: " << noise << 
                             ", seed: " << seed << '\n';

                __runOneSimulation(T, noise, seed);
            }
}

void Simulation::__runOneSimulation(double T, 
                                    double noise,
                                    int seed)
{
    Field phi(__config.Nx,
              __config.Ny,
              __config.phi0,
              noise,
              seed);

    OrderParameter op(std::move(phi),
                      false,    // is_conserved
                      __config.M,
                      __config.kappa,
                      false    // use_FFT
                    );

    LandauPotential potential(
            __config.a0,
            __config.b,
            __config.T0);

    FDSolver solver(phi.nx(),
                    phi.ny());

    for (int step = 0; step < __config.steps; ++step)
    {
        solver.step(op, potential,
                    T, __config.dt);

        // if (shouldSave(step))
        //     save(op, ...);
    }
}