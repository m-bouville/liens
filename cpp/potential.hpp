#pragma once

// #include <vector>

#include "field.hpp"




class Potential
{
public:
    virtual double bulk_energy(const OrderParameter& phi,
                                     double          T)      const = 0;
    
    virtual void   derivative(const OrderParameter& phi,
                                    double          T,
                                    Field&          result) const = 0;
};



class LandauPotential : public Potential
{
public:
    LandauPotential(double a0,
                    double b,
                    double T0)
        : __a0(a0),
          __b(b),
          __T0(T0)
    {}
    
    double bulk_energy(const OrderParameter& order_para,
                             double          T) const override;

    // derivative w.r.t. order parameter phi
    void derivative(const OrderParameter& order_para,
                          double          T,
                          Field&          result) const override;
    
    double minimum(double T) const;

private:
    double __a0;
    double __b;
    double __T0;

    double __a(double T) const;

    double __energy_one_pixel(double phi,
                              double T) const;

    // derivative w.r.t. order parameter phi
    double __derivative_one_pixel(double phi,
                                  double T) const;

};


