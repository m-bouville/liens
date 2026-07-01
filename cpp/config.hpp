#pragma once

#include <string>
#include <vector>

struct Config
{
    // Grid
    int    Nx    = 0;
    int    Ny    = 0;

    // Time integration
    double dt    = 0.0;
    int    steps = 0;

    // threads
    int max_threads= 1;


    // Physics
    double kappa = 0.0;
    double M     = 1.0;

    double a0    = 1.0;
    double b     = 1.0;
    double T0    = 0.0;

    double phi0  = 0.0;

    // Parameter sweeps
    std::vector<double> temperatures;
    std::vector<double> noises;
    std::vector<int>    seeds;

    // Snapshot schedule
    std::vector<int> save;

    void load(const std::string& filename);
    void validate() const;
};