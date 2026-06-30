#pragma once

#include <vector>
#include <cassert>

using std::vector;


class Field
{
public:
    Field(int nx, int ny)
        : Nx(nx),
          Ny(ny),
          values(nx * ny, 0.0)
    {}

    Field(int nx, int ny,
          double phi0, 
          double noiseAmplitude,
          int    seed)
        : Nx(nx),
          Ny(ny),
          values(nx * ny)
    {
        initialize(phi0, noiseAmplitude, seed);
    }
    

    vector<double> values;
    

    // geometry
    int nx() const { return Nx; }
    int ny() const { return Ny; }
    inline int size() const { return Nx * Ny; }
    

    // access in 2D
    inline int index(int i, int j) const
    {        
        assert(i >= 0 && i < Nx);
        assert(j >= 0 && j < Ny);
        return j * Nx + i;
    }
    double& operator()(int i, int j)
    {
        return values[index(i, j)];
    }
    const double& operator()(int i, int j) const
    {
        return values[index(i, j)];
    }
    

    // derivatives (gradient, Laplacian)    
    vector<double> gradient_x()   const;
    vector<double> gradient_y()   const;
    vector<double> gradient_sqr() const;
    vector<double> laplacian()    const;


private:

    int Nx;
    int Ny;

    void initialize(double phi0, 
                    double noiseAmplitude,
                    int    seed);
};



struct OrderParameter
{
public:
    Field  value;
    bool   is_conserved;
    double mobility;
};


class State
{
public:

    State(int nx, 
          int ny,
          vector<OrderParameter> non_conserved,
          vector<OrderParameter> conserved);


    // Phase fields
    vector<OrderParameter> non_conserved;
    vector<OrderParameter> conserved;

};