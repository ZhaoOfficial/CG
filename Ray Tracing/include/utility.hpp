#ifndef _UTILITY_HPP_
#define _UTILITY_HPP_

#include <cmath>
#include <limits>
#include <random>

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * 0.01745329251994329547f;
}

inline float gaussian_float(float mu = 0.0f, float sigma = 0.5f) {
    static std::random_device rd;
    // static std::default_random_engine rng(rd());
    static std::default_random_engine rng;
    std::normal_distribution<float> gaussian(mu, sigma);
    return gaussian(rng);
}

inline float uniform_float(float a = 0.0f, float b = 1.0f) {
    static std::random_device rd;
    // static std::default_random_engine rng(rd());
    static std::default_random_engine rng;
    std::uniform_real_distribution<float> uniform(a, b);
    return uniform(rng);
}

inline int uniform_int(int a = 0, int b = 1) {
    static std::random_device rd;
    // static std::default_random_engine rng(rd());
    static std::default_random_engine rng;
    std::uniform_int_distribution<int> uniform(a, b);
    return uniform(rng);
}

#endif // !_UTILITY_HPP_