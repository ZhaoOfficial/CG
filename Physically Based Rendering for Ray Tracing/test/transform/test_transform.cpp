#include <cassert>
#include <iostream>

#include "debug.h"
#include "transform/transform.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Transform **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Transform`: " << sizeof(Transform) << std::endl;
            assert(sizeof(Transform) == 32 * sizeof(Float));
            assert(sizeof(Transform) == 2 * sizeof(SquareMatrix<4>));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            SquareMatrix<4> mat{};
            SquareMatrix<4> inv_mat{};
            std::array<Float, 16> std_arr_1d{};
            std::array<std::array<Float, 4>, 4> std_arr{};

            Transform T_1;
            Transform T_2{};
            debugOutput(T_1);
            debugOutput(T_1.getInverseMatrix());
            assert(T_1.isIdentity());
        }
        std::cout << "3. Operator overloading\n";
        {
        }
        std::cout << "4. Auxiliary functions\n";
        {
            auto T_1 = translate(Vector3f{1.5, 2.5, 3.5});
            auto T_2 = scale(Vector3f{-1.5, -2.5, -3.5});
            auto T_3 = rotateX(deg2rad(30));
            auto T_4 = rotateY(deg2rad(30));
            auto T_5 = rotateZ(deg2rad(30));

            debugOutput(T_1);
            debugOutput(T_2);
            debugOutput(T_3);
            debugOutput(T_4);
            debugOutput(T_5);
        }
    }

    return 0;
}
