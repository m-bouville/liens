#pragma once

#include <vector>

class Grid
{
public:

    Grid(int nx, int ny);

    int nx() const { return Nx; }
    int ny() const { return Ny; }

    inline int size() const { return Nx * Ny; }
    
    inline int index(int i, int j) const
    {
        return j * Nx + i;
    }

    // Phase field
    std::vector<double> phi;

    // Workspace for nonlinear term
    std::vector<double> nonlinear;

    // Squared wave number
    std::vector<double> k2;

private:

    int Nx;
    int Ny;

    void computeWaveNumbers();
};