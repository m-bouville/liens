#include "Config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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
        throw std::runtime_error("Cannot open configuration file.");

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
            throw std::runtime_error("Invalid line: " + line);

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        seen[key] = true;

        if (key == "Nx")
            Nx = std::stoi(value);

        else if (key == "Ny")
            Ny = std::stoi(value);

        else if (key == "dt")
            dt = std::stod(value);

        else if (key == "steps")
            steps = std::stoi(value);

        else if (key == "kappa")
            kappa = std::stod(value);

        else if (key == "a0")
            a0 = std::stod(value);

        else if (key == "b")
            b = std::stod(value);

        else if (key == "M")
            M = std::stod(value);

        else if (key == "T0")
            T0 = std::stod(value);

        else if (key == "phi0")
            phi0 = std::stod(value);

        else if (key == "temperatures")
        {
            temperatures.clear();

            for (const auto& s : split(value))
                temperatures.push_back(std::stod(s));
        }

        else if (key == "noise")
        {
            noise.clear();

            for (const auto& s : split(value))
                noise.push_back(std::stod(s));
        }

        else if (key == "seeds")
        {
            seeds.clear();

            for (const auto& s : split(value))
                seeds.push_back(std::stod(s));
        }

        else if (key == "save")
        {
            save.clear();

            for (const auto& s : split(value))
                save.push_back(std::stoi(s));
        }

        else
        {
            throw std::runtime_error("Unknown key: " + key);
        }
    }

    const char* required[] =
    {
        "Nx","Ny",
        "dt","steps",
        "kappa","a0","b","M", "T0",
        "phi0",
        "temperatures", "noise", "seeds",
        "save"
    };

    for (auto key : required)
    {
        if (!seen[key])
            throw std::runtime_error(
                std::string("Missing parameter: ") + key);
    }
}