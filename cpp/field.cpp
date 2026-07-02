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
    double min = __values[0];
    double max = __values[0];
    double total     = 0.0;
    double total_sqr = 0.0;  // std deviation

    for (double v : __values)
    {
        if (v < min) min = v;
        if (v > max) max = v;
        total     += v;
        total_sqr += v*v;
    }

    double avg     = total    / __values.size();
    double avg_sqr = total_sqr/ __values.size();

    return {
        {"min", min},
        {"avg", avg},
        {"max", max},
        {"std", std::sqrt(avg_sqr - avg*avg)}  // std deviation
    };
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
    cv::Mat img(__nx, __ny, CV_64F, __values.data());  // TODO __ny, __nx?

    cv::Mat norm;
    cv::normalize(img, norm, -127.5, 127.5, cv::NORM_MINMAX);

    norm.convertTo(norm, CV_8U);

    cv::imwrite(file.string(), norm);
}

