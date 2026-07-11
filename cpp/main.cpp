#include <iostream>
#include <filesystem>

#include "config.hpp"
#include "snapshot_writer.hpp"
#include "simulation.hpp"



int main(int argc, char* argv[])
{

    try
    {
        // Rebuild datasets/*/metadata.txt for backward compatibility.
        writer::rebuild_dataset_metadata("../datasets");

        std::cout << "Starting..." << '\n';

        // Loading config file
        Config cfg;

        std::string filename = argc > 1 ? argv[1] : "config.txt";
        std::cout << "Loading " << filename << "...\n";
        cfg.load(filename);

        std::cout << "Validating...\n";
        cfg.validate();

        std::cout << "dimensions: "     << cfg.Nx << " x " << cfg.Ny << '\n';
        std::cout << cfg.temperatures.size() << " temperatures * " << 
                     cfg.noises.size() << " noises * " <<                      
                     cfg.seeds.size() << " seeds = " <<                    
                     cfg.temperatures.size() * cfg.seeds.size() * cfg.noises.size() << " simulations.\n"; 


        // run several simulations
        std::cout << "Starting " << cfg.max_threads << " simulations.\n";
        Simulation sim(cfg);
        sim.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
    }
}
