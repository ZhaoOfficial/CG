#ifndef _PERLIN_HPP_
#define _PERLIN_HPP_

#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    class Perlin {
    public:
        Perlin();
        ~Perlin();

        float noise(const Point3& p) const;
        float turbulence(const Point3& p, int depth = 7) const;
    
    private:
        static const int point_count = 256;
        Vec3* rand_vec;
        int* permute_x;
        int* permute_y;
        int* permute_z;

        static int* perlinGeneratePermutation();
        static float trillinearInterpolation(Vec3 near[2][2][2], float u, float v, float w);
    };
}

#endif // !_PERLIN_HPP_
