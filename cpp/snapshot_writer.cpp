#include "snapshot_writer.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>   // setw, fixed, setprecision

#include <cmath>
#include <numbers>   // for pi

#include <mutex>

#include "finite_differences.hpp"
#include "statistics.hpp"



namespace fs = std::filesystem;


// Saving file
uint16_t writer::float_to_half(float f)
{
    uint32_t x = std::bit_cast<uint32_t>(f);

    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x007FFFFF;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;

    if (exp <= 0)
        return sign; // underflow → 0

    if (exp >= 31)
        return sign | 0x7C00; // inf

    return sign | (exp << 10) | (mantissa >> 13);
}

void writer::save_phi_half(const std::vector<double>& phi,
                            const fs::path& file)
{
    std::ofstream out(file, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Failed to open file: " + file.string());
    }

    for (double v : phi)
    {
        float f = static_cast<float>(v);
        uint16_t h = float_to_half(f);
        out.write(reinterpret_cast<char*>(&h), sizeof(uint16_t));
    }
}


std::string writer::make_dir_name(int nx, int ny,
                                  double T, double noise, int seed)
{
    auto fmt = [](double x, int scale)
    {
        int v = std::round(x * scale);
        return v;
    };

    int Ti = fmt(T,     1000);   // 0.950 -> 950
    int ni = fmt(noise, 1000);   // 0.05  ->  50

    std::ostringstream oss;
    oss << "../datasets/"
        << nx << "x"  << ny << "/"
        << "T" << Ti
        << "_n" << std::setw(3) << std::setfill('0') << ni
        << "_s" << seed;

    return oss.str();
}



// Calculating and displaying statistics
writer::WriterStatistics writer::statistics(const OrderParameter& op, 
                                                  std::ostringstream& log, 
                                                  int             step,
                                                  double          T, 
                                            const Potential&      potential, 
                                            const Solver&         solver, 
                                            const FD::Neighbors&  neighbors)
{
    WriterStatistics out;

    out.time_step = step;

    // average and std deviation for phi
    auto stats = op.statistics();
    out.avg_phi   = stats.average;
    out.stdev_phi = stats.stdev;
    
    // fraction for phi below or above some threshold
    auto stats_frac = op.phase_fractions({-0.1, 0.}, {0.1});
    out.phi_below_neg10= stats_frac.below[0];
    out.phi_below_0    = stats_frac.below[1];
    out.phi_above_10   = stats_frac.above[0];

    // gradient
    out.avg_gradient = FD::avg_gradient(op.field(), neighbors);
    out.gradient_sqr = FD::gradient_sqr(op.field(), neighbors) / op.size();

    // characteristic length
    int max_dist = std::min(op.nx()*2/3, op.ny()*2/3);
            // to cut computation time
    // std::cout << "max_dist for autocorrelation: " << max_dist << '\n';

    
    static std::mutex fftw_mutex;
    {
        std::lock_guard<std::mutex> lock(fftw_mutex);
        
        auto autocorr = statistics::autocorrelation(op.field());  // , max_dist);
        out.autocorr_length = autocorr.peak_distance;
        out.autocorr_correl = autocorr.peak_value;
    }

    // anisotropy
    auto aniso = statistics::structure_tensor(op.field(), neighbors);
    // out.trace      = aniso.trace;        // λ1+λ2
    out.anisotropy = aniso.anisotropy;   // (λ1-λ2)/(λ1+λ2)
    out.angle      = aniso.angle;        // radians in [-π/2, π/2]

    
    // energy
    out.energy = solver.energy(op, potential, neighbors, T);

    // display statistics to console
    log << std::left
        << std::setw(8) << step
        << std::setw(7) << std::fixed << std::setprecision(1)
        << stats.min * 100 
        << std::setw(7) << stats.average * 100 
        << std::setw(7) << stats.max * 100 
        << std::setw(7) << stats.stdev * 100 
        << std::setw(7) << out.avg_gradient*100 << std::setw(7) << out.gradient_sqr*100
        << std::setw(7) << std::setprecision(2) << out.autocorr_correl*100
        << std::setw(4) << out.autocorr_length
        // << std::setw(7) << out.trace*100   
        << std::setw(7) << out.anisotropy*100   
        << std::setw(7) << std::setprecision(0) << out.angle*180/std::numbers::pi
        << std::setw(8) << std::setprecision(1) << out.energy << '\n';

    // reset (important if more printing follows)
    log.unsetf(std::ios::fixed);
    log << std::setprecision(6);

    return out;
}

void writer::write_csv(const std::filesystem::path&         filename,
                       const std::vector<WriterStatistics>& stats)
{
    std::ofstream out(filename);

    out << "step,avg_phi,stdev_phi,"
           "phi_below_-10,phi_below_0,phi_above_10,"
           "avg_gradient,gradient_sqr,"
           "autocorr_correl,autocorr_length,"
           "anisotropy,angle,"
           "energy\n";

    for (const auto& s : stats)
        out << s.time_step      << ','
            << s.avg_phi        << ',' << s.stdev_phi      << ','
            << s.phi_below_neg10<< ',' << s.phi_below_0    << ','<< s.phi_above_10<< ','
            << s.avg_gradient   << ',' << s.gradient_sqr   << ','
            << s.autocorr_correl<< ',' << s.autocorr_length<< ','
            << s.anisotropy     << ',' << s.angle      << ','
            << s.energy         << '\n';
}

void writer::write_metadata(const std::filesystem::path& file,
                            const Config& config,
                            double      T,
                            double      noise,
                            int         seed,
                            std::string code_version,
                            bool        completed)
{
    std::ofstream out(file);
    if (!out)
        throw std::runtime_error("Failed to open " + file.string());

    out << "# Phase-field simulation metadata\n\n";

    out << "directory    = " << file.parent_path().generic_string() << '\n';
    out << "code version = " << code_version << '\n';
    out << "status       = " << (completed ? "complete" : "incomplete") << '\n';
    out << '\n';

    out << "# Grid\n";
    out << "Nx           = " << config.Nx << '\n';
    out << "Ny           = " << config.Ny << '\n';
    out << '\n';

    out << "# Time integration\n";
    out << "dt           = " << config.dt << '\n';
    out << "steps        = " << config.steps << '\n';
    out << "save_steps   =";
    for (int s : config.save)
        if (s <= config.steps)
            out << ' ' << s;
    out << '\n';
    out << '\n';

    out << "# Landau potential \n";
    out << "a0           = " << config.a0 << "   # quadratic term in Landau potential\n";
    out << "b            = " << config.b  << "   # fourth-degree term in Landau potential\n";
    out << "T0           = " << config.T0 << "   # threshold temperature in Landau potential\n";
    out << '\n';

    out << "# Physical parameters \n";
    out << "temperature  = " << T << '\n';
    out << "kappa        = " << config.kappa << "   # coefficient of gradient in free energy\n";
    out << "mobility     = " << config.M << '\n';
    out << '\n';
    
    out << "# Initialization \n";
    out << "phi0         = " << config.phi0  << "   # initial average value of the order parameter\n";
    out << "noise        = " << noise << '\n';
    out << "seed         = " << seed << '\n';
    out << '\n';

    out << "# Equations \n";
    out << "equation     = Allen-Cahn\n";       // TODO hardcoded for now
    out << "solver       = finite difference\n";// TODO hardcoded for now
    out << '\n';
}