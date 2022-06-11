#include <cassert>
#include <cmath>
#include <iostream>

#include "common.h"
#include "geometry/vector2.h"
#include "geometry/vector3.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Vector2 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Vector2f`: " << sizeof(Vector2f) << std::endl;
            assert(sizeof(Vector2f) == 2 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Vector2f vec2_1;
            Vector2f vec2_2 = vec2_1;
            Vector2f vec2_3{ 1, 2 };
            std::cout << "vec2_1 = " << vec2_1 << std::endl;
            std::cout << "vec2_2 = " << vec2_2 << std::endl;
            std::cout << "vec2_3 = " << vec2_3 << std::endl;
            assert(vec2_1.x == Float(0) and vec2_1.y == Float(0));
            assert(vec2_2.x == Float(0) and vec2_2.y == Float(0));
            assert(vec2_3.x == Float(1) and vec2_3.y == Float(2));

            vec2_2 = vec2_3;
            std::cout << "vec2_2 = " << vec2_2 << std::endl;
            assert(vec2_3.x == Float(1) and vec2_3.y == Float(2));

        }
        std::cout << "3. Operator overloading\n";
        {
            Vector2f vec2_1;
            Vector2f vec2_2{ 1, 2.5 };
            auto vec2_3 = vec2_1 + vec2_2;
            auto vec2_4 = vec2_1 - vec2_2;
            auto vec2_5 = vec2_2 * 2;
            auto vec2_6 = 2 * vec2_2;
            auto vec2_7 = vec2_2 / 2;
            std::cout << "vec2_3 = " << vec2_3 << std::endl;
            std::cout << "vec2_4 = " << vec2_4 << std::endl;
            std::cout << "vec2_5 = " << vec2_5 << std::endl;
            std::cout << "vec2_6 = " << vec2_6 << std::endl;
            std::cout << "vec2_7 = " << vec2_7 << std::endl;
            assert((vec2_1 != Vector2f{1, 2.5}));
            assert((vec2_3 == Vector2f{1, 2.5}));
            assert((vec2_4 == -Vector2f{1, 2.5}));
            assert((vec2_5 == Vector2f{2, 5}));
            assert((vec2_6 == Vector2f{2, 5}));
            assert((vec2_7 == Vector2f{0.5, 1.25}));

            // Indexing operators
            for (std::size_t i{}; i < 2; ++i) {
                std::cout << vec2_7.data[i] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "4. Auxiliary functions\n";
        {
            auto vec2_1 = Vector2f::Zeros();
            auto vec2_2 = Vector2f::Ones();
            std::cout << "vec2_2.squareNorm() = " << vec2_2.squareNorm() << std::endl;
            std::cout << "vec2_2.norm() = " << vec2_2.norm() << std::endl;
            assert(vec2_1.x == Float(0) and vec2_1.y == Float(0));
            assert(vec2_2.x == Float(1) and vec2_2.y == Float(1));
            assert(vec2_2.squareNorm() == Float(2));
            assert(vec2_2.norm() == std::sqrt(Float(2)));

            Vector2f vec2_3 = Vector2f{ 1.0, 0.25 };
            Vector2f vec2_4 = Vector2f{ 2.0, -0.5 };
            std::cout << "vec2_3.min() = " << vec2_3.min() << std::endl;
            std::cout << "vec2_3.max() = " << vec2_3.max() << std::endl;
            assert(vec2_3.min() == Float(0.25));
            assert(vec2_3.max() == Float(1.0));

            auto min_vec2 = vec2_3.cwiseMin(vec2_4);
            auto max_vec2 = vec2_3.cwiseMax(vec2_4);
            auto prod_vec2 = vec2_3.cwiseProd(vec2_4);
            auto div_vec2 = vec2_3.cwiseDiv(vec2_4);
            auto dot_res  = dot(vec2_3, vec2_4);
            auto abs_vec2 = abs(vec2_4);
            auto normalized_vec2 = normalized(Vector2f{3, 4});
            std::cout << "min_vec2 = " << min_vec2 << std::endl;
            std::cout << "max_vec2 = " << max_vec2 << std::endl;
            std::cout << "prod_vec2 = " << prod_vec2 << std::endl;
            std::cout << "div_vec2 = " << div_vec2 << std::endl;
            std::cout << "dot_res = " << dot_res << std::endl;
            std::cout << "abs_vec2 = " << abs_vec2 << std::endl;
            std::cout << "normalized_vec2 = " << normalized_vec2 << std::endl;
            assert((min_vec2 == Vector2f{1.0, -0.5}));
            assert((max_vec2 == Vector2f{2.0, 0.25}));
            assert((prod_vec2 == Vector2f{2.0, -0.125}));
            assert((div_vec2 == Vector2f{0.5, -0.5}));
            assert(dot_res == Float(1.0 * 2.0 - 0.25 * 0.5));
            assert((abs_vec2 == Vector2f{2.0, 0.5}));
            assert((normalized_vec2 == Vector2f{0.6, 0.8}));
        }
    }

    std::cout << "********** Vector3 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Vector3f`: " << sizeof(Vector3f) << std::endl;
            assert(sizeof(Vector3f) == 3 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Vector3f vec3_1;
            Vector3f vec3_2 = vec3_1;
            Vector3f vec3_3{ 1, 2, 3 };
            std::cout << "vec3_1 = " << vec3_1 << std::endl;
            std::cout << "vec3_2 = " << vec3_2 << std::endl;
            std::cout << "vec3_3 = " << vec3_3 << std::endl;
            assert(vec3_1.x == Float(0) and vec3_1.y == Float(0) and vec3_1.z == Float(0));
            assert(vec3_2.x == Float(0) and vec3_2.y == Float(0) and vec3_2.z == Float(0));
            assert(vec3_3.x == Float(1) and vec3_3.y == Float(2) and vec3_3.z == Float(3));

            vec3_2 = vec3_3;
            std::cout << "vec3_2 = " << vec3_2 << std::endl;
            assert(vec3_3.x == Float(1) and vec3_3.y == Float(2) and vec3_3.z == Float(3));

        }
        std::cout << "3. Operator overloading\n";
        {
            Vector3f vec3_1;
            Vector3f vec3_2{ 1, 2.5, 4 };
            auto vec3_3 = vec3_1 + vec3_2;
            auto vec3_4 = vec3_1 - vec3_2;
            auto vec3_5 = vec3_2 * 2;
            auto vec3_6 = 2 * vec3_2;
            auto vec3_7 = vec3_2 / 2;
            std::cout << "vec3_3 = " << vec3_3 << std::endl;
            std::cout << "vec3_4 = " << vec3_4 << std::endl;
            std::cout << "vec3_5 = " << vec3_5 << std::endl;
            std::cout << "vec3_6 = " << vec3_6 << std::endl;
            std::cout << "vec3_7 = " << vec3_7 << std::endl;
            assert((vec3_1 != Vector3f{1, 2.5, 4}));
            assert((vec3_3 == Vector3f{1, 2.5, 4}));
            assert((vec3_4 == -Vector3f{1, 2.5, 4}));
            assert((vec3_5 == Vector3f{2, 5, 8}));
            assert((vec3_6 == Vector3f{2, 5, 8}));
            assert((vec3_7 == Vector3f{0.5, 1.25, 2}));

            // Indexing operators
            for (std::size_t i{}; i < 3; ++i) {
                std::cout << vec3_7.data[i] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "4. Auxiliary functions\n";
        {
            auto vec3_1 = Vector3f::Zeros();
            auto vec3_2 = Vector3f::Ones();
            std::cout << "vec3_2.squareNorm() = " << vec3_2.squareNorm() << std::endl;
            std::cout << "vec3_2.norm() = " << vec3_2.norm() << std::endl;
            assert(vec3_1.x == Float(0) and vec3_1.y == Float(0) and vec3_1.z == Float(0));
            assert(vec3_2.x == Float(1) and vec3_2.y == Float(1) and vec3_2.z == Float(1));
            assert(vec3_2.squareNorm() == Float(3));
            assert(vec3_2.norm() == std::sqrt(Float(3)));

            Vector3f vec3_3 = Vector3f{ 1.0, 0.25, 2.0 };
            Vector3f vec3_4 = Vector3f{ 2.0, -0.5, -1.0 };
            std::cout << "vec3_3.min() = " << vec3_3.min() << std::endl;
            std::cout << "vec3_3.max() = " << vec3_3.max() << std::endl;
            assert(vec3_3.min() == Float(0.25));
            assert(vec3_3.max() == Float(2.0));

            auto min_vec3 = vec3_3.cwiseMin(vec3_4);
            auto max_vec3 = vec3_3.cwiseMax(vec3_4);
            auto prod_vec3 = vec3_3.cwiseProd(vec3_4);
            auto div_vec3 = vec3_3.cwiseDiv(vec3_4);
            auto dot_res  = dot(vec3_3, vec3_4);
            auto abs_vec3 = abs(vec3_4);
            auto normalized_vec3 = normalized(Vector3f{2, 3, 6});
            auto cross_vec3 = cross(Vector3f{1, 0, 0}, Vector3f{0, 1, 0});
            std::cout << "min_vec3 = " << min_vec3 << std::endl;
            std::cout << "max_vec3 = " << max_vec3 << std::endl;
            std::cout << "prod_vec3 = " << prod_vec3 << std::endl;
            std::cout << "div_vec3 = " << div_vec3 << std::endl;
            std::cout << "dot_res = " << dot_res << std::endl;
            std::cout << "abs_vec3 = " << abs_vec3 << std::endl;
            std::cout << "normalized_vec3 = " << normalized_vec3 << std::endl;
            std::cout << "cross_vec3 = " << cross_vec3 << std::endl;
            assert((min_vec3 == Vector3f{1.0, -0.5, -1.0}));
            assert((max_vec3 == Vector3f{2.0, 0.25, 2.0}));
            assert((prod_vec3 == Vector3f{2.0, -0.125, -2.0}));
            assert((div_vec3 == Vector3f{0.5, -0.5, -2.0}));
            assert(dot_res == Float(1.0 * 2.0 - 0.25 * 0.5 - 1.0 * 2.0));
            assert((abs_vec3 == Vector3f{2.0, 0.5, 1.0}));
            assert((normalized_vec3 == (Vector3f{2, 3, 6} / Float(7))));
            assert((cross_vec3 == Vector3f{0, 0, 1}));
        }
    }

    return 0;
}
