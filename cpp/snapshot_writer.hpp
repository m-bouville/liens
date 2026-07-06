
#include <bit>
#include <vector>
#include <filesystem>
#include <sstream>   // std::ostringstream

#include "config.hpp"
#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"



namespace writer 
{        
    struct WriterStatistics
    {
        int    time_step;

        double avg_phi;
        double stdev_phi;

        double phi_below_neg10;
        double phi_below_0;
        double phi_above_10;

        double avg_gradient;
        double gradient_sqr;

        double autocorr_correl;
        int    autocorr_length;

        // double trace;          // identical to gradient_sqr
        double anisotropy;
        double angle;           // in radians

        double energy;
    };

    uint16_t float_to_half(float f);

    void save_phi_half(const std::vector<double>& phi,
                       const std::filesystem::path& file);

    std::string make_dir_name(int nx, int ny,
                              double T, double noise, int seed);

                              
    WriterStatistics statistics(const OrderParameter& op, std::ostringstream& log,
                                int step, double T, 
                                const Potential& potential, const Solver& solver, 
                                const FD::Neighbors& neighbors);

    void     write_csv(const std::filesystem::path&         filename,
                       const std::vector<WriterStatistics>& stats);

    void write_metadata(const std::filesystem::path& filename,
                        const Config& config,
                              double      T,
                              double      noise,
                              int         seed,
                              std::string code_version,
                              bool        completed = false);
    

                              
    // Write metadata file for each system size

    // Called once before launching a sweep for one grid size.
    // Creates datasets/128x128/metadata.txt if it does not exist.
    void create_dataset_metadata(
        const std::filesystem::path& dataset_dir,
        int nx,
        int ny,
        const std::vector<double>& temperatures,
        const std::vector<double>& noises,
        const std::vector<int>&    seeds);

    // Called after a run successfully finishes.
    void register_completed_run(const std::filesystem::path& run_dir);

    
    // Rebuild datasets/*/metadata.txt by scanning the filesystem.
    // Safe to call at startup for backward compatibility.
    void rebuild_dataset_metadata(
        const std::filesystem::path& datasets_root = "../datasets");
}