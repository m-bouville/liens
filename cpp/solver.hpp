#pragma once

#include  <memory>  // for unique_ptr

#include "finite_differences.hpp"  // namespace FD
#include "potential.hpp"


class Solver
{
public:
    virtual void step(      OrderParameter& op,
                      const Potential&      potential, 
                      const FD::Neighbors&  neighbors,
                            double          T,
                            double          dt) = 0;

    virtual double energy(const OrderParameter& op,
                          const Potential&      potential, 
                          const FD::Neighbors&  neighbors,
                                double          T) const = 0;
};


class FDSolver : public Solver
{
public:
    FDSolver(int nx, 
             int ny)
        : __ws(nx, ny)
    {}

    void step(      OrderParameter& op,
              const Potential&      potential, 
              const FD::Neighbors&  neighbors,
                    double          T,
                    double          dt) override;


    double energy(const OrderParameter& op,
                  const Potential&      potential, 
                  const FD::Neighbors&  neighbors,
                        double          T) const override;

private:
    FD::Workspace __ws;  // storing derivatives
};