
#include <bit>
#include <vector>
#include <filesystem>
#include <sstream>   // std::ostringstream

#include "field.hpp"
#include "potential.hpp"
#include "solver.hpp"

struct Statistics
{
    int    time_step;
    double avg_phi;
    double stdev_phi;
    double phi_below_neg10;
    double phi_below_0;
    double phi_above_10;
    double avg_gradient;
    double gradient_sqr;
    double energy;
};


namespace writer 
{
    uint16_t float_to_half(float f);

    void save_phi_half(const std::vector<double>& phi,
                       const std::filesystem::path& file);

    std::string make_dir_name(int nx, int ny,
                              double T, double noise, int seed);

                              
    Statistics statistics(const OrderParameter& op, std::ostringstream& log,
                          int step, double T, 
                          const Potential& potential, const Solver& solver);

    void     write_csv(const std::filesystem::path&   filename,
                       const std::vector<Statistics>& stats);
}