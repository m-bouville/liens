#pragma once

#include "field.hpp"
#include "finite_differences.hpp"


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
    