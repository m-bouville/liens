#include "snapshot_writer.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>   // setw, fixed, setprecision

#include <cmath>



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
Statistics writer::statistics(const OrderParameter& op, 
                                    std::ostringstream& log, 
                                    int             step,
                                    double          T, 
                              const Potential&      potential, 
                              const Solver&         solver)
{
    Statistics out;

    out.time_step = step;

    // average and std deviation for phi
    auto stats = op.statistics();
    out.avg_phi   = stats["avg"];
    out.stdev_phi = stats["std"];
    
    // fraction for phi below or above some threshold
    auto stats_frac = op.phase_fractions({-0.1, 0.}, {0.1});
    out.phi_below_neg10= stats_frac.below[0];
    out.phi_below_0    = stats_frac.below[1];
    out.phi_above_10   = stats_frac.above[0];

    // energy
    out.energy = solver.energy(op, potential, T);

    // gradient
    out.avg_gradient = FD::avg_gradient(op.field());
    out.gradient_sqr = FD::gradient_sqr(op.field());

    // TODO characteristic length

    // TODO anisotropy


    // display statistics to console
    log << std::left
        << std::setw(8) << step
        << std::setw(7) << std::fixed << std::setprecision(1)
        << stats["min"]* 100 
        << std::setw(7) << stats["avg"]* 100 
        << std::setw(7) << stats["max"]* 100 
        << std::setw(7) << stats["std"]* 100 
        << std::setw(8) << std::setprecision(1) << out.energy << '\n';

    // reset (important if more printing follows)
    log.unsetf(std::ios::fixed);
    log << std::setprecision(6);

    return out;
}

void writer::write_csv(const std::filesystem::path&   filename,
                       const std::vector<Statistics>& stats)
{
    std::ofstream out(filename);

    out << "step,avg_phi,stdev_phi,"
           "phi_below_-10,phi_below_0,phi_above_10,"
           "avg_gradient,gradient_sqr,"
           "energy\n";

    for (const auto& s : stats)
        out << s.time_step    << ','
            << s.avg_phi      << ','
            << s.stdev_phi    << ','
            << s.phi_below_neg10<< ','
            << s.phi_below_0  << ','
            << s.phi_above_10 << ','
            << s.avg_gradient << ','
            << s.gradient_sqr << ','
            << s.energy << '\n';
}