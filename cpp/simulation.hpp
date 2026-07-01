#pragma once

#include "config.hpp"


class Simulation
{
public:

    Simulation(const Config& config)
      : __config(config)
    {}

    void run();

private:

    Config __config;

    void __runOneSimulation(double T,
                            double noise,
                            int seed);


    // SnapshotWriter writer;
};