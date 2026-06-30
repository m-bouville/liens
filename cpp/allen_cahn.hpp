#include <iostream>
#include <vector>
#include <complex>

#include "config.hpp"
#include "state.hpp"
#include "field.hpp"


class EvolutionEquation
{
public:
    virtual ~EvolutionEquation() = default;

    virtual const vector<OrderParameter>& order_parameters() const = 0; 
};



class AllenCahnEquation : public EvolutionEquation
{
public:

    AllenCahnEquation(const State&  state,
                      const Config& cfg,
                      double temperature);

    const vector<OrderParameter>& order_parameters() const override 
    { return state.non_conserved; }

private:

    //-----------------------------
    // Physical parameters
    //-----------------------------

    double M;
    double kappa;

    //-----------------------------
    // State
    //-----------------------------
    const State& state;


    //-----------------------------
    // Internal routines
    //-----------------------------
};