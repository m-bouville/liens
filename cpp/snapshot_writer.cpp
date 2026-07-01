#include "snapshot_writer.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

#include "field.hpp"


namespace fs = std::filesystem;


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