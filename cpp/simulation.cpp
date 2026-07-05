#include "simulation.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <algorithm>
#include <unordered_set>
#include <cstdint>

#include <future>
// #include "BS_thread_pool.hpp"

#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"
#include "snapshot_writer.hpp"   // writer namespace

// BS::thread_pool pool(6);


void Simulation::run()
{
    const std::size_t max_threads = __config.max_threads;
    
    std::vector<std::future<void>> futures;

    // std::cout <<  __config.save << '\n';

    for (int seed : __config.seeds)
        for (double noise : __config.noises)
            for (double T : __config.temperatures)
            {
                futures.push_back(
                    std::async(std::launch::async,
                        &Simulation::__runOneSimulation,
                        this, T, noise, seed)
                );

                if (futures.size() >= max_threads)
                {
                    futures.front().get();
                    futures.erase(futures.begin());
                }

                // pool.submit_task([=, this] {
                    // __runOneSimulation(T, noise, seed);
                // }
            }

    for (auto &f : futures)
        f.get();
}

void Simulation::__runOneSimulation(double T, 
                                    double noise,
                                    int    seed)
{
    std::filesystem::path outdir = 
            writer::make_dir_name(__config.Nx, __config.Ny, T, noise, seed);

    std::ostringstream os_start;
    if (std::filesystem::exists(outdir / "COMPLETE"))
    {
        os_start << "Skipping completed run for " << 
                "T: " << T << ", noise: " << noise << ", seed: " << seed << '\n';
        return;
    }
    else 
    {
        os_start << "Starting run for " << "T: " << T << ", noise: " << noise << 
                ", seed: " << seed << '\n';
    }
        
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << os_start.str();
    }
  
    std::filesystem::create_directories(outdir);
    
    writer::write_metadata(outdir / "metadata.txt", __config,
                            T, noise, seed,
                            "2026-07-03",
                            false);

    Field phi(__config.Nx,
              __config.Ny,
              __config.phi0,
              noise,
              seed);
    
    const FD::Neighbors neighbors(__config.Nx, __config.Ny);

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


    std::unordered_set<int> save_steps(__config.save.begin(),
                                       __config.save.end  ());


    std::ostringstream log;
    log << '\n' << "T: " << T << ", noise: " << noise << 
                ", seed: " << seed << '\n';
    log << "minimum at " << potential.minimum(T) << '\n';
    log << std::left
            << std::setw( 8) << "step"
            << std::setw(28) << "   order parameter (%)"
            << std::setw(14) << "  gradient"
            << std::setw(11) << "autocorrel"
            << std::setw(11) << "   anisotropy"
            << std::setw( 8) << "total"
            << '\n';  
    log << std::left
            << std::setw( 8) << " "
            << std::setw( 7) << "min" << std::setw( 7) << "avg" << std::setw( 7) << "max"
            << std::setw( 7) << "stdev"
            << std::setw( 7) << "<|g|>"   << std::setw( 7) << "<g^2>"
            << std::setw( 7) << "value" << std::setw( 4) << "at"
            // << std::setw( 7) << "trace" << std::setw( 7)  // identical to <g^2>
            << "diff" << std::setw( 4) << "ang"
            << std::setw( 8) << "energy"
            << '\n';          
    log << std::string(90, '-') << '\n';


    std::filesystem::path file;
    std::vector<writer::WriterStatistics> all_stats;

    for (int step = 0; step < __config.steps; ++step)
    {
        solver.step(op, potential, neighbors,
                    T, __config.dt);

        if (save_steps.contains(step))
        {
            // display
            all_stats.push_back(
                writer::statistics(op, log, step, T, potential, solver, neighbors)
            );


            // save
            std::ostringstream name;
            name << "t" << std::setw(7) << std::setfill('0') << step;

            file = outdir / name.str();
            // log << "Saving to " << file.string() << "...\n";    


            writer::save_phi_half(op.phi(), file);
        }
    }
    std::ofstream(outdir / "COMPLETE").close();
    writer::write_metadata(outdir / "metadata.txt", __config,
                            T, noise, seed,
                            "2026-07-03",
                            true);

    op.save_as_png(file.replace_extension(".png"));
    writer::write_csv(outdir / "statistics.csv", all_stats);

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << log.str() << '\n';  
    }
}
