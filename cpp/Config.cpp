#include "config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

using std::runtime_error;
using std::to_string;
using std::stod;   using std::stoi;

namespace
{

std::string trim(std::string s)
{
    auto left = s.find_first_not_of(" \t\r\n");
    if (left == std::string::npos)
        return "";

    auto right = s.find_last_not_of(" \t\r\n");

    return s.substr(left, right - left + 1);
}

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;

    std::stringstream ss(s);

    std::string item;

    while (std::getline(ss, item, ','))
        out.push_back(trim(item));

    return out;
}

}

void Config::load(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw runtime_error("Cannot open configuration file.");

    std::unordered_map<std::string, bool> seen;

    std::string line;

    while (std::getline(file, line))
    {
        line = trim(line);

        if (line.empty())
            continue;

        if (line[0] == '#')
            continue;

        auto pos = line.find('=');

        if (pos == std::string::npos)
            throw runtime_error("Invalid line: " + line);

        std::string key   = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        seen[key] = true;

        if (key == "Nx")
            Nx = stoi(value);

        else if (key == "Ny")
            Ny = stoi(value);

        else if (key == "dt")
            dt = stod(value);

        else if (key == "steps")
            steps = stoi(value);

        else if (key == "kappa")
            kappa = stod(value);

        else if (key == "a0")
            a0 = stod(value);

        else if (key == "b")
            b = stod(value);

        else if (key == "M")
            M = stod(value);

        else if (key == "T0")
            T0 = stod(value);

        else if (key == "phi0")
            phi0 = stod(value);

        else if (key == "temperatures")
        {
            temperatures.clear();

            for (const auto& s : split(value))
                temperatures.push_back(stod(s));
        }

        else if (key == "noises")
        {
            noises.clear();

            for (const auto& s : split(value))
                noises.push_back(stod(s));
        }

        else if (key == "seeds")
        {
            seeds.clear();

            for (const auto& s : split(value))
                seeds.push_back(stoi(s));
        }

        else if (key == "save")
        {
            save.clear();

            for (const auto& s : split(value))
                save.push_back(stoi(s));
        }

        else
        {
            throw runtime_error("Unknown key: " + key);
        }
    }

    const char* required[] =
    {
        "Nx","Ny",
        "dt","steps",
        "kappa", "a0", "b", "M", "T0",
        "phi0",
        "temperatures", "noises", "seeds",
        "save"
    };

    for (auto key : required)
    {
        if (!seen[key])
            throw runtime_error(
                std::string("Missing parameter: ") + key);
    }
}

bool is_power_of_two(unsigned int n)
{
    return n != 0 && (n & (n - 1)) == 0;
}

void Config::validate() const
{
    // Grid dimensions
    if (Nx <= 0)
        throw runtime_error("Nx must be positive, not " + to_string(Nx));
    if (Ny <= 0)
        throw runtime_error("Ny must be positive, not " + to_string(Ny));
    if (Nx != Ny)
        std::cerr << "Nx and Ny should be equal.\n";
    if (!is_power_of_two(Nx))
        std::cerr << "Nx should be a power of 2.\n";
    if (!is_power_of_two(Ny))
        std::cerr << "Ny should be a power of 2.\n";

    // Time integration
    if (dt <= 0)
        throw runtime_error("dt must be positive, not " + to_string(dt));
    if (steps <= 0)
        throw runtime_error("steps must be positive, not " + steps);
    
    // Physics
    if (kappa <= 0)
        throw runtime_error("kappa must be positive, not " + to_string(kappa));
    if (a0 <= 0)
        throw runtime_error("a0 must be positive, not " + to_string(a0));
    if (b <= 0)
        throw runtime_error("b must be positive, not " + to_string(b));
    if (M <= 0)
        throw runtime_error("M must be positive, not " + to_string(M));
    if (T0 <= 0)
        throw runtime_error("T0 must be positive, not " + to_string(T0));
}