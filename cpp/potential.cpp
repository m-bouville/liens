
#include "potential.hpp"

#include <vector>
#include <cmath>

#include "field.hpp"


double LandauPotential::__a(double T) const 
{ return __a0 * (T - __T0); }


// Energy
double LandauPotential::__energy_one_pixel(double phi,
                                           double a) const
{
    double phi2 = phi*phi;
    return a/2. * phi2 + __b/4. * phi2*phi2;
}

double LandauPotential::bulk_energy(const OrderParameter& order_para,
                                          double T) const
{
    const std::vector<double>& values = order_para.values();

    double result = 0;
    const double a = __a(T);

    for (size_t i = 0; i < values.size(); ++i)
        result += __energy_one_pixel(values[i], a);

    return result;
}


// derivative w.r.t. order parameter phi
// double LandauPotential::__derivative_one_pixel(double phi,
//                                                 double a) const
// { return a * phi + __b * phi*phi*phi; }

void LandauPotential::derivative(const OrderParameter& order_para,
                                       double          T,
                                       Field&          result) const
{
    const std::vector<double>& values = order_para.values();
    const double a = __a(T);

    
    // vectorization-friendly
    const double* __restrict in = values.data();
    double* __restrict out = result.values().data();

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const double phi = in[i];
        out[i] = a * phi + __b * phi * phi * phi;
    }

    // legible
    // for (size_t i = 0; i < values.size(); ++i)
    //     result.values()[i] = __derivative_one_pixel(values[i], a);
}


double LandauPotential::minimum(double T) const
{
    if (T >= __T0)
        return 0.;
    return std::sqrt(-__a(T) / __b);
}