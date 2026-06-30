
#include <vector>

#include <state.hpp>


using std::vector;


class FreeEnergy
{
public:

    virtual void derivative(
        const State& state,
        vector<Field>& dF) const = 0;
};


class LandauGinzburg : public FreeEnergy
{
public:



    //-----------------------------
    // Physical parameters
    //-----------------------------
    
    double M;
    double kappa;
};








class Potential
{
public:
    virtual double energy    (const OrderParameter& phi) const = 0;
    
    virtual void   derivative(const OrderParameter& phi,
                                    vector<double>& result) const = 0;
};



class LandauPotential : public Potential
{
public:

    double a0;
    double b;
    double T0;
    double T;

    double a() const
        { return a0 * (T - T0); }

    double energy_one_pixel(double phi) const
    {
        double phi2 = phi*phi;
        return a()/2. * phi2 + b/4. * phi2*phi2;
    }
    double energy(const OrderParameter& phi) const override
    {
        const vector<double>& values = phi.value.values;

        double result = 0;

        for (size_t i = 0; i < values.size(); ++i)
            result += energy_one_pixel(values[i]);

        return result;
    }

    // derivative w.r.t. order parameter phi
    double derivative_one_pixel(double phi) const
    {
        return a() * phi + b * phi*phi*phi;
    }
    void derivative(const OrderParameter& phi,
                              vector<double>& result) const override
    {
        const vector<double>& values = phi.value.values;

        result.resize(values.size());

        for (size_t i = 0; i < values.size(); ++i)
            result[i] = derivative_one_pixel(values[i]);
    }
};


