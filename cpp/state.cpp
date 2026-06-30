#include "state.hpp"

#include <vector>
#include <cmath>
#include <random>

using std::vector;



void Field::initialize(double phi0, 
                       double noiseAmplitude,
                       int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-noiseAmplitude/2,
                                                 noiseAmplitude/2);

    // values.resize(size());

    for (double& v : values)
        v = phi0 + dist(rng);
}



// derivatives (gradient, Laplacian)    
vector<double> Field::gradient_x() const
{
    vector<double> gx(size());

    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            int ip = (i + 1) % Nx;
            int im = (i - 1 + Nx) % Nx;

            gx[index(i, j)] =
                0.5 * ((*this)(ip, j) - (*this)(im, j));
        }
    }

    return gx;
}

vector<double> Field::gradient_y() const
{
    vector<double> gy(size());

    for (int j = 0; j < Ny; ++j)
    {
        int jp = (j + 1) % Ny;
        int jm = (j - 1 + Ny) % Ny;

        for (int i = 0; i < Nx; ++i)
        {
            gy[index(i, j)] =
                0.5 * ((*this)(i, jp) - (*this)(i, jm));
        }
    }

    return gy;
}

vector<double> Field::gradient_sqr() const
{
    auto gx = gradient_x();
    auto gy = gradient_y();

    vector<double> result(size());

    for (int k = 0; k < size(); ++k)
        result[k] = gx[k] * gx[k] + gy[k] * gy[k];

    return result;
}

vector<double> Field::laplacian() const
{
    vector<double> lap(size());

    for (int j = 0; j < Ny; ++j)
    {
        int jp = (j + 1) % Ny;
        int jm = (j - 1 + Ny) % Ny;

        for (int i = 0; i < Nx; ++i)
        {
            int ip = (i + 1) % Nx;
            int im = (i - 1 + Nx) % Nx;

            lap[index(i, j)] =
                (*this)(ip, j) +
                (*this)(im, j) +
                (*this)(i, jp) +
                (*this)(i, jm) -
                4.0 * (*this)(i, j);
        }
    }

    return lap;
}




class State
{
public:
    State(int nx,
          int ny,
          vector<OrderParameter> non_conserved,
          vector<OrderParameter> conserved)
        : non_conserved(std::move(non_conserved)),
          conserved(std::move(conserved))
    {}

    vector<OrderParameter> non_conserved;
    vector<OrderParameter> conserved;
};


