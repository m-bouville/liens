#include "field.hpp"

#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

using std::vector;



void Field::initialize(double phi0, 
                       double noiseAmplitude,
                       int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-noiseAmplitude/2,
                                                 noiseAmplitude/2);

    // values.resize(size());

    for (double& v : __values)
        v = phi0 + dist(rng);
}


// statistics
double Field::average() const
{
    double total = 0.;

    for (double v : __values)
        total += v;
    return total / size();
}

std::map<std::string, double> Field::statistics() const
{
    if (__values.empty())
        throw std::runtime_error("Field is empty");

    double min = __values[0];
    double max = __values[0];
    double total = 0.0;

    for (double v : __values)
    {
        if (v < min) min = v;
        if (v > max) max = v;
        total += v;
    }

    return {
        {"min", min},
        {"avg", total / __values.size()},
        {"max", max}
    };
}


// save as image
void Field::save_as_png(const std::filesystem::path& file)
{
    cv::Mat img(__nx, __ny, CV_64F, __values.data());  // TODO __ny, __nx?

    cv::Mat norm;
    cv::normalize(img, norm, -127.5, 127.5, cv::NORM_MINMAX);

    norm.convertTo(norm, CV_8U);

    cv::imwrite(file.string(), norm);
}

