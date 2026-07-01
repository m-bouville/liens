#include <vector>
#include <cmath>
#include <random>

#include "field.hpp"

using std::vector;



void Field::initialize(double phi0, 
                       double noiseAmplitude,
                       int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-noiseAmplitude/2,
                                                 noiseAmplitude/2);

    // values.resize(size());

    for (double& v : __values)
        v = phi0 + dist(rng);
}




