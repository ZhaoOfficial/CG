#ifndef _UTILITY_HPP_
#define _UTILITY_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <functional>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;
using std::fabs;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * 0.01745329251994329547f;
}

inline float gaussian_float(float mu = 0.0f, float sigma = 0.5f) {
    std::normal_distribution<float> gaussian(mu, sigma);
    static std::default_random_engine rng;
    return gaussian(rng);
}

inline float uniform_float(float a = 0.0f, float b = 1.0f) {
    std::uniform_real_distribution<float> uniform(a, b);
    static std::default_random_engine rng;
    return uniform(rng);
}

#endif // !_UTILITY_HPP_