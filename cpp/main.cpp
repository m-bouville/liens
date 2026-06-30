#include <iostream>

#include "config.hpp"
#include "state.hpp"
// #include "allen_cahn.hpp"
// #include "free_energy.hpp"
// #include "simulation.hpp"



int main(int argc, char* argv[])
{
    // Loading config file
    Config cfg;

    cfg.load(argc > 1 ? argv[1] : "../config.txt");
    cfg.validate();

    std::cout << "dimensions: "     << cfg.Nx << " x " << cfg.Ny << '\n';
    std::cout << "nb temperatures: "<< cfg.temperatures.size() << '\n';


    
    for (double temperature : cfg.temperatures)
        for (double noise : cfg.noises)
            for (int seed : cfg.seeds)
            {
                std::cout << "T: " << temperature << ", noise: " << noise << ", seed: " << seed << '\n';
                // creating one non-conserved order parameter
                
                OrderParameter phi{
                    Field(cfg.Nx, cfg.Ny, cfg.phi0, noise, seed),
                    false,  // is_conserved
                    cfg.M
                };
                // std::cout << phi.size() << '\n';


                struct RunParameters
                {
                    double temperature;
                    double noise;
                    int seed;
                };

                RunParameters run{temperature, noise, seed};


                // Simulation sim(cfg, run);
                // sim.run();
            }
}

// double df(double phi, double T) {
//     return a(T)*phi + b*phi*phi*phi;
// }