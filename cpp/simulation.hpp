#pragma once

#include "config.hpp"

#include <mutex>


class Simulation
{
public:

    Simulation(const Config& config)
      : __config(config)
    {}

    void run();

private:

    Config __config;

    std::mutex cout_mutex;

    bool __stop_requested() const;

    void __runOneSimulation(double T,
                            double noise,
                            int seed);


    // SnapshotWriter writer;
};