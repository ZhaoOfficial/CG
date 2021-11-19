#include "perlin.hpp"
#include "utility.hpp"

namespace RayTracing {

    Perlin::Perlin() {
        this->rand_vec = new Vec3[Perlin::point_count];
        for (int i = 0; i < Perlin::point_count; ++i) {
            this->rand_vec[i] = random_in_unit_cube(-1.0f, 1.0f);
        }

        this->permute_x = Perlin::perlinGeneratePermutation();
        this->permute_y = Perlin::perlinGeneratePermutation();
        this->permute_z = Perlin::perlinGeneratePermutation();
    }

    Perlin::~Perlin() {
        delete[] this->rand_vec;
        delete[] this->permute_x;
        delete[] this->permute_y;
        delete[] this->permute_z;
    }

    float Perlin::noise(const Point3& p) const {
        float u = p[0] - std::floor(p[0]);
        float v = p[1] - std::floor(p[1]);
        float w = p[2] - std::floor(p[2]);

        int i = static_cast<int>(std::floor(p[0]));
        int j = static_cast<int>(std::floor(p[1]));
        int k = static_cast<int>(std::floor(p[2]));

        Vec3 near[2][2][2];
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    near[a][b][c] = this->rand_vec[
                        permute_x[(i + a) & 0xff] ^ permute_y[(j + b) & 0xff] ^ permute_z[(k + c) & 0xff]
                    ];
                }
            }
        }

        return trillinearInterpolation(near, u, v, w);
    }

    float Perlin::turbulence(const Point3& p, int depth) const {
        float accumulation = 0.0f;
        Point3 temp_p = p;
        float weight = 1.0f;

        for (int i = 0; i < depth; ++i) {
            accumulation += weight * noise(temp_p);
            weight *= 0.5f;
            temp_p *= 2.0f;
        }

        return std::fabs(accumulation);
    }

    int* Perlin::perlinGeneratePermutation() {
        int* p = new int[Perlin::point_count];
        for (int i = 0; i < Perlin::point_count; ++i) {
            p[i] = i;
        }
        
        for (int i = Perlin::point_count - 1; i > 0; --i) {
            int target = uniform_int(0, i);
            std::swap(p[i], p[target]);
        }

        return p;
    }

    float Perlin::trillinearInterpolation(Vec3 near[2][2][2], float u, float v, float w) {
        
        float uu = u * u * (3.0f - 2 * u);
        float vv = v * v * (3.0f - 2 * v);
        float ww = w * w * (3.0f - 2 * w);
        
        float accumulation = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    Vec3 weight(u - i, v - j, w - k);
                    accumulation += (i * uu + (1 - i) * (1 - uu))
                                  * (j * vv + (1 - j) * (1 - vv))
                                  * (k * ww + (1 - k) * (1 - ww))
                                  * dot(near[i][j][k], weight);
                }
            }
        }
        return accumulation;
    }

}
