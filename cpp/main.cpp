
#include "Config.hpp"
#include "Grid.hpp"

#include <iostream>

int main(int argc, char* argv[])
{
    // Loading config file
    Config cfg;

    cfg.load(argc > 1 ? argv[1] : "../config.txt");

    std::cout << "dimensions: " << cfg.Nx << " x " << cfg.Ny << '\n';
    std::cout << "nb temperatures: " << cfg.temperatures.size() << '\n';



    Grid grid(cfg.Nx, cfg.Ny);

    std::cout << grid.size() << '\n';
    std::cout << "k²: " << grid.k2[0] << " x " << grid.k2[1] << '\n';
    std::cout << grid.k2[cfg.Nx] << '\n';
}

// double df(double phi, double T) {
//     return a(T)*phi + b*phi*phi*phi;
// }