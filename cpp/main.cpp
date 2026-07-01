#include <iostream>

#include "config.hpp"
#include "simulation.hpp"



int main(int argc, char* argv[])
{
    // Loading config file
    Config cfg;

    cfg.load(argc > 1 ? argv[1] : "../config.txt");
    cfg.validate();

    std::cout << "dimensions: "     << cfg.Nx << " x " << cfg.Ny << '\n';
    std::cout << "nb temperatures: "<< cfg.temperatures.size() << '\n';


    // run several simulations
    Simulation sim(cfg);
    sim.run();
    
}
