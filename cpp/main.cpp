#include <iostream>
#include <filesystem>

#include "config.hpp"
#include "snapshot_writer.hpp"
#include "simulation.hpp"



int main(int argc, char* argv[])
{

    try
    {
        // std::cout << "Starting..." << '\n';

        // Rebuild datasets/*/metadata.txt for backward compatibility.
        writer::rebuild_dataset_metadata("../datasets");
        // std::cout << "writer::rebuild_dataset_metadata(\'../datasets\')\n";

        // Loading config file
        Config cfg;

        std::string filename = argc > 1 ? argv[1] : "config.txt";
        std::cout << "Loading " << filename << "...\n";
        cfg.load(filename);

        // std::cout << "Validating...\n";
        cfg.validate();

        
        // create directory before running `write_dataset_metadata`
        std::ostringstream dataset_dir;
        dataset_dir << "../datasets/" << cfg.Nx << "x"  << cfg.Ny << "/";
        std::filesystem::create_directories(dataset_dir.str());
        writer::create_dataset_metadata(
            dataset_dir.str(), cfg.Nx, cfg.Ny, cfg.temperatures, cfg.noises, cfg.seeds);


        std::cout << "dimensions: "     << cfg.Nx << " x " << cfg.Ny << '\n';
        std::cout << cfg.temperatures.size() << " temperatures * " << 
                     cfg.noises.size() << " noises * " <<                      
                     cfg.seeds.size() << " seeds = " <<                    
                     cfg.temperatures.size() * cfg.seeds.size() * cfg.noises.size() << 
                     " simulations in total.\n";

        // run several simulations
        Simulation sim(cfg);
        sim.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
    }
}
