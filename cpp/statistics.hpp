#pragma once

#include "field.hpp"
#include "finite_differences.hpp"

// struct Statistics
// {
//     double avg_phi;
//     double stdev_phi;
//     double phi_below_10;
//     double phi_below_50;
//     double phi_above_90;
//     double avg_gradient;
// };

namespace statistics
{
    // Anisotropy
    struct StructureTensor
    {
        double Jxx_avg;
        double Jxy_avg;
        double Jyy_avg;

        double lambda1;      // largest eigenvalue
        double lambda2;      // smallest eigenvalue

        double trace;        //  λ1 + λ2
        double anisotropy;   // (λ1-λ2) / (λ1+λ2)
        double angle;        // radians in [-π/2, π/2]
    };

    StructureTensor structure_tensor(const Field& phi, const FD::Neighbors& nb);


    // characteristic length
    struct AutoCorrelMetrics
    {
        std::vector<double> values;

        int peak_distance;
        double peak_value;
    };

    AutoCorrelMetrics autocorrelation(const Field& phi);  // , int max_dist);
}
    