#include "Grid.hpp"

#include <cmath>

Grid::Grid(int nx, int ny)
    : Nx(nx),
      Ny(ny),
      phi(nx * ny),
      nonlinear(nx * ny),
      k2(nx * ny)
{
    computeWaveNumbers();
}


void Grid::computeWaveNumbers()
{
    for (int j = 0; j < Ny; ++j)
    {
        int ky = (j <= Ny / 2) ? j : j - Ny;

        for (int i = 0; i < Nx; ++i)
        {
            int kx = (i <= Nx / 2) ? i : i - Nx;

            k2[j * Nx + i] =
                static_cast<double>(kx * kx + ky * ky);
        }
    }
}
