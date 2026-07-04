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

FieldStatistics Field::statistics() const
{
    FieldStatistics out;

    double min  = __values[0];
    double max  = __values[0];
    double mean = 0.0;
    double M2   = 0.0;

    std::size_t n = 0;
    double delta;

    for (double v : __values)
    {
        ++n;
        if (v < min) min = v;
        if (v > max) max = v;
        
    // Welford's algorithm
        delta = v - mean;
        mean += delta / n;
        M2 += delta * (v - mean);
    }

    out.average    = mean;
    double variance= std::max(0.0, M2 / n);
    out.stdev      = std::sqrt(variance);  // std deviation

    out.min = min;
    out.max = max;

    return out;
}


ThresholdFractions Field::phase_fractions(
    std::initializer_list<double> below,
    std::initializer_list<double> above) const
{
    std::vector<std::size_t> nb_below(below.size(), 0);
    std::vector<std::size_t> nb_above(above.size(), 0);

    for (double v : __values)
    {
        std::size_t i = 0;
        for (double t : below)
        {
            if (v < t)
                ++nb_below[i];
            ++i;
        }

        i = 0;
        for (double t : above)
        {
            if (v > t)
                ++nb_above[i];
            ++i;
        }
    }

    const double invN = 1.0 / __values.size();

    ThresholdFractions result;
    result.below.resize(below.size());
    result.above.resize(above.size());

    for (std::size_t i = 0; i < below.size(); ++i)
        result.below[i] = nb_below[i] * invN;

    for (std::size_t i = 0; i < above.size(); ++i)
        result.above[i] = nb_above[i] * invN;

    return result;
}


// save as image
void Field::save_as_png(const std::filesystem::path& file)
{
    cv::Mat img(__ny, __nx, CV_64F, __values.data());  // TODO __ny, __nx?

    cv::Mat norm;
    cv::normalize(img, norm, 0.0, 255.0, cv::NORM_MINMAX);

    norm.convertTo(norm, CV_8U);

    cv::imwrite(file.string(), norm);
}

