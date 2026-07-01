#include "simulation.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

#include <algorithm>
#include <unordered_set>
#include <cstdint>

#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"
#include "snapshot_writer.hpp"   // writer namespace




void Simulation::run()
{
    
    // std::cout <<  __config.save << '\n';

    for (int seed : __config.seeds)
        for (double noise : __config.noises)
            for (double T : __config.temperatures)
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
    
    std::filesystem::path outdir = writer::make_dir_name(T, noise, seed);

    if (std::filesystem::exists(outdir / "COMPLETE"))
    {
        std::cout << "Skipping completed run: " << outdir.string() << "\n";
        return;
    }

std::filesystem::create_directories(outdir);

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
    std::cout << "minimum at " << potential.minimum(T) << '\n';

    FDSolver solver(phi.nx(),
                    phi.ny());


    std::unordered_set<int> save_steps(__config.save.begin(),
                                       __config.save.end  ());
    std::cout << std::left
            << std::setw(10) << "step"
            << std::setw(15) << "avg (%)"
            << std::setw(15) << "energy"
            << '\n';          
    std::cout << std::string(40, '-') << '\n';

    std::filesystem::path file;

    for (int step = 0; step < __config.steps; ++step)
    {
        solver.step(op, potential,
                    T, __config.dt);

        if (save_steps.contains(step))
        {
            // display
            std::cout << std::left
                    << std::setw(10) << step
                    << std::setw(15) << std::fixed << std::setprecision(6)
                    << op.average() * 100
                    << std::setw(15)
                    << solver.energy(op, potential, T)
                    << '\n';

            // reset (important if more printing follows)
            std::cout.unsetf(std::ios::fixed);
            std::cout << std::setprecision(6);

            // save
            std::ostringstream name;
            name << "t" << std::setw(7) << std::setfill('0') << step;

            file = outdir / name.str();
            // std::cout << "Saving to " << file.string() << "...\n";    


            writer::save_phi_half(op.phi(), file);
        }
    }
    std::ofstream(outdir / "COMPLETE").close();


    auto [minIt, maxIt] = std::minmax_element(op.phi().begin(), op.phi().end());
    std::cout << "min = " << *minIt
            << " max = " << *maxIt << "\n";

    op.save_as_png(file.replace_extension(".png"));
}
