#include <cassert>

#include "debug.h"
#include "math/square_matrix.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Square Matrix, N = 1 **********\n";
    {
        std::cout << "1. Size\n";
        {
            Float data_1d[1] = {1.5};
            SquareMatrix<1> M1_1{data_1d};
            std::cout << "Size of `SquareMatrix<1>` = " << sizeof(M1_1) << std::endl;
            assert(sizeof(M1_1) == sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Float data_1d[1] = {1.5};
            Float data[1][1] = {{2.5}};
            SquareMatrix<1> M1_1{data_1d};
            SquareMatrix<1> M1_2{data};

            debugOutput(M1_1);
            debugOutput(M1_2);
            //! M1_1[{0, 0}] is ok.
            debugOutput(M1_1[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_2[(Vector2<std::size_t> {0, 0})]);
            assert(M1_1[(Vector2<std::size_t> {0, 0})] == Float(1.5));
            assert(M1_2[(Vector2<std::size_t> {0, 0})] == Float(2.5));
        }
        std::cout << "3. Operator overloading\n";
        {
            Float data_1d[1] = {1.5};
            Float data[1][1] = {{2.5}};
            SquareMatrix<1> M1_1{data_1d};
            SquareMatrix<1> M1_2{data};

            auto M1_3 = M1_1 + M1_2;
            auto M1_4 = M1_1 - M1_2;
            auto M1_5 = M1_1 * Float(2);
            auto M1_6 = Float(2) * M1_1;
            auto M1_7 = M1_1 / Float(2);

            debugOutput(M1_3);
            debugOutput(M1_4);
            debugOutput(M1_5);
            debugOutput(M1_6);
            debugOutput(M1_7);
            debugOutput(M1_3[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_4[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_5[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_6[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_7[(Vector2<std::size_t> {0, 0})]);

            assert(M1_3[(Vector2<std::size_t> {0, 0})] == Float(4.0));
            assert(M1_4[(Vector2<std::size_t> {0, 0})] == Float(-1.0));
            assert(M1_5[(Vector2<std::size_t> {0, 0})] == Float(3.0));
            assert(M1_6[(Vector2<std::size_t> {0, 0})] == Float(3.0));
            assert(M1_7[(Vector2<std::size_t> {0, 0})] == Float(0.75));
            assert(M1_1 != M1_2);
            assert(M1_1 < M1_2);
            assert(!(M1_1 < M1_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            Float data[1][1] = {{1}};
            SquareMatrix<1> M1_1;
            SquareMatrix<1> M1_2 = SquareMatrix<1>::Zeros();
            SquareMatrix<1> M1_3{data};
            Float det1 = M1_1.determinant();
            Float det2 = M1_3.determinant();
            SquareMatrix<1> M1_4 = M1_3.inverse();

            debugOutput(M1_1);
            debugOutput(M1_2);
            debugOutput(M1_3);
            debugOutput(M1_2.isIdentity());
            debugOutput(M1_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M1_4);

            assert(M1_1 == M1_2);
            assert(M1_3.isIdentity());
            assert(det1 == Float(0));
            assert(det2 == Float(1));
            assert(M1_4 == M1_3);
        }
    }

    std::cout << "********** Square Matrix, N = 3 **********\n";
    {
        std::cout << "1. Size\n";
        {
            Float data_1d[9] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
            SquareMatrix<3> M3_1{data_1d};
            std::cout << "Size of `SquareMatrix<3>` = " << sizeof(M3_1) << std::endl;
            assert(sizeof(M3_1) == 9 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Float data_1d[9] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
            Float data[1][1] = {{2.5}};
            SquareMatrix<3> M3_1{data_1d};
            SquareMatrix<1> M1_2{data};

            debugOutput(M3_1);
            debugOutput(M1_2);
            debugOutput(M3_1[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_2[(Vector2<std::size_t> {0, 0})]);
            assert(M3_1[(Vector2<std::size_t> {0, 0})] == Float(1.5));
            assert(M1_2[(Vector2<std::size_t> {0, 0})] == Float(2.5));
        }
        std::cout << "3. Operator overloading\n";
        {
            Float data_1d[1] = {1.5};
            Float data[1][1] = {{2.5}};
            SquareMatrix<1> M1_1{data_1d};
            SquareMatrix<1> M1_2{data};

            auto M1_3 = M1_1 + M1_2;
            auto M1_4 = M1_1 - M1_2;
            auto M1_5 = M1_1 * Float(2);
            auto M1_6 = Float(2) * M1_1;
            auto M1_7 = M1_1 / Float(2);

            debugOutput(M1_3);
            debugOutput(M1_4);
            debugOutput(M1_5);
            debugOutput(M1_6);
            debugOutput(M1_7);
            debugOutput(M1_3[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_4[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_5[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_6[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_7[(Vector2<std::size_t> {0, 0})]);

            assert(M1_3[(Vector2<std::size_t> {0, 0})] == Float(4.0));
            assert(M1_4[(Vector2<std::size_t> {0, 0})] == Float(-1.0));
            assert(M1_5[(Vector2<std::size_t> {0, 0})] == Float(3.0));
            assert(M1_6[(Vector2<std::size_t> {0, 0})] == Float(3.0));
            assert(M1_7[(Vector2<std::size_t> {0, 0})] == Float(0.75));
            assert(M1_1 != M1_2);
            assert(M1_1 < M1_2);
            assert(!(M1_1 < M1_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            Float data[1][1] = {{1}};
            SquareMatrix<1> M1_1;
            SquareMatrix<1> M1_2 = SquareMatrix<1>::Zeros();
            SquareMatrix<1> M1_3{data};
            Float det1 = M1_1.determinant();
            Float det2 = M1_3.determinant();
            SquareMatrix<1> M1_4 = M1_3.inverse();

            debugOutput(M1_1);
            debugOutput(M1_2);
            debugOutput(M1_3);
            debugOutput(M1_2.isIdentity());
            debugOutput(M1_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M1_4);

            assert(M1_1 == M1_2);
            assert(M1_3.isIdentity());
            assert(det1 == Float(0));
            assert(det2 == Float(1));
            assert(M1_4 == M1_3);
        }
    }

    return 0;
}
