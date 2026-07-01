
#include <bit>
#include <vector>
#include <filesystem>


namespace writer 
{
    uint16_t float_to_half(float f);

    void save_phi_half(const std::vector<double>& phi,
                       const std::filesystem::path& file);

    std::string make_dir_name(int nx, int ny,
                              double T, double noise, int seed);

}