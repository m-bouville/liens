
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

        double trace;
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
}