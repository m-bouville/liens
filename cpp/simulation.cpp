#include "simulation.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <algorithm>
#include <unordered_set>
#include <cstdint>

#include <chrono>
#include <ctime>

#include <future>
// #include "BS_thread_pool.hpp"

#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"
#include "snapshot_writer.hpp"   // writer namespace

// BS::thread_pool pool(6);

struct SimulationParameters
{
    double T;
    double noise;
    int    seed;
};


void Simulation::run()
{
    const std::size_t max_threads = __config.max_threads;
    
    std::vector<std::future<void>> futures;

    // std::cout <<  __config.save << '\n';


    // How many simulations are (in)complete?
    std::size_t nb_done     = 0;
    std::size_t nb_remaining= 0;
    std::vector<SimulationParameters> simulations_to_run;


    for (int seed : __config.seeds)
        for (double noise : __config.noises)
            for (double T : __config.temperatures)
            {
                std::filesystem::path outdir = writer::make_dir_name(
                        __config.Nx, __config.Ny, T, noise, seed);

                if (!std::filesystem::exists(outdir / "COMPLETE")) {
                    ++nb_remaining;
                    simulations_to_run.push_back({T, noise, seed});
                    // std::cout << "Not done yet: " << outdir << '\n';
                } else
                    ++nb_done;
            }

    std::cout << nb_done << " simulations done, " << nb_remaining << " to run.\n";
    std::cout << "Running on " << __config.max_threads << " threads.\n";
    
    // run what needs to be run
    for (const auto& sim : simulations_to_run)
    {
        futures.push_back(
            std::async(std::launch::async,
                &Simulation::__runOneSimulation,
                this,
                sim.T,
                sim.noise,
                sim.seed)
        );

        if (futures.size() >= max_threads)
        {
            futures.front().get();
            futures.erase(futures.begin());
        }
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

    
    std::ostringstream para_display;
    para_display << "Run for {T:"  << std::setw(6) << T << 
           ", noise:" << std::setw(6) << noise << 
           ", seed:"  << std::setw(4) << seed << '}';

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        std::cout << para_display.str() << " starting at " << 
                std::put_time(std::localtime(&now_c), "%H:%M") << '\n';
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
    // log << '\n';
    log << para_display.str();
        //   << " (minimum at " << potential.minimum(T)*100 << "%)" << '\n';

    // // header (on two lines)
    // log << std::left
    //         << std::setw( 8) << "step"
    //         << std::setw(21) << "   order parameter (%)"
    //         << std::setw(14) << "  gradient"
    //         << std::setw(11) << "autocorrel"
    //         << std::setw(11) << "   anisotropy"
    //         << std::setw( 8) << "total"
    //         << '\n';  
    // log << std::left
    //         << std::setw( 8) << " "
    //         << std::setw( 7) << "min"   << std::setw( 7) << "avg" << std::setw( 7) << "max"
    //         << std::setw( 7) << "stdev"
    //         << std::setw( 7) << "<|g|>" << std::setw( 7) << "<g^2>"
    //         << std::setw( 7) << "value" << std::setw( 4) << "at"
    //         // << std::setw( 7) << "trace" // identical to <g^2>
    //         << std::setw( 7) << "diff" << std::setw( 4)  << "ang"
    //         << std::setw( 8) << "energy"
    //         << '\n';          
    // log << std::string(90, '-') << '\n';


    std::filesystem::path file;
    std::vector<writer::WriterStatistics> all_stats;

    for (int step = 0; step < __config.steps; ++step)
    {
        solver.step(op, potential, neighbors, T, __config.dt);

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

    writer::register_completed_run(outdir);

    {        
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << log.str() << " finished at " << 
                std::put_time(std::localtime(&now_c), "%H:%M") << '\n';  // << '\n';  
    }
}
