#pragma once

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <stdexcept>
#include <filesystem>



class Field
{
public:

    // constructors
    Field(int nx, int ny)
        : __nx(nx),
          __ny(ny),
          __values(nx * ny, 0.0)
    {}

    Field(int nx, int ny,
          double phi0, 
          double noiseAmplitude,
          int    seed)
        : __nx(nx),
          __ny(ny),
          __values(nx * ny)
    {
        initialize(phi0, noiseAmplitude, seed);
    }
            

    // grid dimensions
    int nx() const { return __nx; }
    int ny() const { return __ny; }
    inline int size() const { return __nx * __ny; }
    

    // accessors: whole grid
    std::vector<double>& values()
    { return __values; }
    const std::vector<double>& values() const
    { return __values; }

    // accessors: 1D index
    double& operator[](int idx)
    { return __values[idx]; }
    const double operator[](int idx) const
    { return __values[idx]; }


    // accessors: 2D grid coordinates
    inline int index(int i, int j) const
    {        
        assert(i >= 0 && i < __nx);
        assert(j >= 0 && j < __ny);
        return j * __nx + i;
    }
    double& operator()(int i, int j)
    { return __values[index(i, j)]; }
    const double operator()(int i, int j) const
    { return __values[index(i, j)]; }

    // statistics
    double average() const;
    std::map<std::string, double> statistics() const;

    // save as image
    void save_as_png(const std::filesystem::path& file);
    

private:
    int __nx;
    int __ny;

    std::vector<double> __values;

    void initialize(double phi0, 
                    double noiseAmplitude,
                    int    seed);
};




class OrderParameter
{
public:
    OrderParameter(Field&& phi,
                   bool    is_conserved,
                   double  mobility,
                   double  kappa,
                   bool    use_FFT)
          : __phi         (std::move(phi)),
            __is_conserved(is_conserved),
            __mobility    (mobility),
            __kappa       (kappa),
            __use_FFT     (use_FFT)
    {
        if (use_FFT) 
            throw std::invalid_argument("Fourier solver not implemented.");
    }

    // read-only accessors: grid dimensions
    int nx()   const { return __phi.nx(); }
    int ny()   const { return __phi.ny(); }
    int size() const { return __phi.size(); }
        
    // read-only accessors: parameters
    bool   is_conserved() const { return __is_conserved; }
    double mobility()     const { return __mobility; } 
    double kappa()        const { return __kappa; } 
    bool   use_FFT()      const { return __use_FFT; }

    // read-only accessors
    const Field& field() const { return __phi; }
    const std::vector<double>& values() const 
    { return __phi.values(); }
    const std::vector<double>& phi() const   // synonym
    { return __phi.values(); }  
    const double operator[](int idx)      const { return __phi[idx]; }
    const double operator()(int i, int j) const { return __phi(i, j);}

    // read-write accessors
    double& operator[](int idx)      { return __phi[idx]; }
    double& operator()(int i, int j) { return __phi(i, j);}
    
    // statistics
    double average   () {return __phi.average   (); }
    auto   statistics() {return __phi.statistics(); }
    
    // save as image
    void save_as_png(const std::filesystem::path& file)
    {return __phi.save_as_png(file); }


private:
    Field  __phi;
    bool   __is_conserved;
    double __mobility;
    double __kappa;
    bool   __use_FFT;

};
