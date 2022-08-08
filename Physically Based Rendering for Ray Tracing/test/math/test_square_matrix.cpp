#include <cassert>

#include "debug.h"
#include "math/square_matrix.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Square Matrix, N = 1 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::array<Float, 1> data_1d = {};
            SquareMatrix<1> M1_1{data_1d};
            std::cout << "Size of `SquareMatrix<1>` = " << sizeof(M1_1) << std::endl;
            assert(sizeof(M1_1) == sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            std::array<Float, 1> data_1d = {1.5};
            std::array<std::array<Float, 1>, 1> data = {{2.5}};
            SquareMatrix<1> M1_1{data_1d};
            SquareMatrix<1> M1_2{data};

            debugOutput(M1_1);
            debugOutput(M1_2);
            //! M1_1[{0, 0}] is ok.
            debugOutput(M1_1[(Vector2<std::size_t> {0, 0})]);
            debugOutput(M1_2[(Vector2<std::size_t> {0, 0})]);
            assert(M1_1[(Vector2<std::size_t> {0, 0})] == Float{1.5});
            assert(M1_2[(Vector2<std::size_t> {0, 0})] == Float{2.5});
        }
        std::cout << "3. Operator overloading\n";
        {
            std::array<Float, 1> data_1d = {1.5};
            std::array<std::array<Float, 1>, 1> data = {{2.5}};
            SquareMatrix<1> M1_1{data_1d};
            SquareMatrix<1> M1_2{data};

            auto M1_3 = M1_1 + M1_2;
            auto M1_4 = M1_1 - M1_2;
            auto M1_5 = M1_1 * Float{2};
            auto M1_6 = Float{2} * M1_1;
            auto M1_7 = M1_1 / Float{2};

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

            assert(M1_3[(Vector2<std::size_t> {0, 0})] == Float{4.0});
            assert(M1_4[(Vector2<std::size_t> {0, 0})] == Float{-1.0});
            assert(M1_5[(Vector2<std::size_t> {0, 0})] == Float{3.0});
            assert(M1_6[(Vector2<std::size_t> {0, 0})] == Float{3.0});
            assert(M1_7[(Vector2<std::size_t> {0, 0})] == Float{0.75});
            assert(M1_1 != M1_2);
            assert(M1_1 < M1_2);
            assert(!(M1_1 < M1_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            std::array<std::array<Float, 1>, 1> data = {{1}};
            SquareMatrix<1> M1_1;
            SquareMatrix<1> M1_2 = SquareMatrix<1>::Zeros();
            SquareMatrix<1> M1_3{data};
            Float det1 = M1_1.determinant();
            Float det2 = M1_3.determinant();
            SquareMatrix<1> M1_4 = M1_3.inverse();
            SquareMatrix<1> M1_5 = M1_3.transpose();

            debugOutput(M1_1);
            debugOutput(M1_2);
            debugOutput(M1_3);
            debugOutput(M1_2.isIdentity());
            debugOutput(M1_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M1_4);
            debugOutput(M1_5);

            assert(M1_1 == M1_2);
            assert(M1_3.isIdentity());
            assert(det1 == Float{0});
            assert(det2 == Float{1});
            assert(M1_4 == M1_3);
            assert(M1_5[(Vector2<std::size_t> {0, 0})] == Float{1});
        }
    }

    std::cout << "********** Square Matrix, N = 2 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::array<Float, 4> data_1d = {};
            SquareMatrix<2> M2_1{data_1d};
            std::cout << "Size of `SquareMatrix<4>` = " << sizeof(M2_1) << std::endl;
            assert(sizeof(M2_1) == 4 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            std::array<Float, 4> data_1d = {1.5, 2.5, 3.5, 4.5};
            std::array<std::array<Float, 2>, 2> data = {
                std::array<Float, 2>{3.25, 2.0}, 
                std::array<Float, 2>{5.0, 4.0}
            };
            SquareMatrix<2> M2_1{data_1d};
            SquareMatrix<2> M2_2{data};

            debugOutput(M2_1);
            debugOutput(M2_2);
            assert(M2_1[(Vector2<std::size_t> {1, 0})] == Float{3.5});
            assert(M2_2[(Vector2<std::size_t> {1, 0})] == Float{5.0});
        }
        std::cout << "3. Operator overloading\n";
        {
            std::array<Float, 4> data_1d = {1.5, 2.5, 3.5, 4.5};
            std::array<std::array<Float, 2>, 2> data = {
                std::array<Float, 2>{3.25, 2.0}, 
                std::array<Float, 2>{5.0, 4.0}
            };
            SquareMatrix<2> M2_1{data_1d};
            SquareMatrix<2> M2_2{data};

            auto M2_3 = M2_1 + M2_2;
            auto M2_4 = M2_1 - M2_2;
            auto M2_5 = M2_1 * Float{2};
            auto M2_6 = Float{2} * M2_1;
            auto M2_7 = M2_1 / Float{2};

            debugOutput(M2_3);
            debugOutput(M2_4);
            debugOutput(M2_5);
            debugOutput(M2_6);
            debugOutput(M2_7);

            assert(M2_3[(Vector2<std::size_t> {0, 1})] == Float{4.5});
            assert(M2_4[(Vector2<std::size_t> {0, 1})] == Float{0.5});
            assert(M2_5[(Vector2<std::size_t> {0, 1})] == Float{5.0});
            assert(M2_6[(Vector2<std::size_t> {0, 1})] == Float{5.0});
            assert(M2_7[(Vector2<std::size_t> {0, 1})] == Float{1.25});
            assert(M2_1 != M2_2);
            assert(M2_1 < M2_2);
            assert(!(M2_2 < M2_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            std::array<std::array<Float, 2>, 2> data = {
                std::array<Float, 2>{3.25, 2.0}, 
                std::array<Float, 2>{5.0, 4.0}
            };
            SquareMatrix<2> M2_1;
            SquareMatrix<2> M2_2 = SquareMatrix<2>::Zeros();
            SquareMatrix<2> M2_3{data};
            Float det1 = M2_1.determinant();
            Float det2 = M2_3.determinant();
            SquareMatrix<2> M2_4 = M2_3.inverse();
            SquareMatrix<2> M2_5 = M2_3.transpose();

            debugOutput(M2_1);
            debugOutput(M2_2);
            debugOutput(M2_3);
            debugOutput(M2_2.isIdentity());
            debugOutput(M2_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M2_4);
            debugOutput(M2_5);

            assert(M2_1 == M2_2);
            assert(not M2_3.isIdentity());
            assert(det1 == Float{0});
            assert(det2 == Float{3});
            assert(M2_4[(Vector2<std::size_t> {0, 0})] == Float{4} / Float{3});
            assert(M2_5[(Vector2<std::size_t> {1, 0})] == Float{2});
        }
    }

    std::cout << "********** Square Matrix, N = 3 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::array<Float, 9> data_1d = {};
            SquareMatrix<3> M3_1{data_1d};
            std::cout << "Size of `SquareMatrix<3>` = " << sizeof(M3_1) << std::endl;
            assert(sizeof(M3_1) == 9 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            std::array<Float, 9> data_1d = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
            std::array<std::array<Float, 3>, 3> data = {
                std::array<Float, 3>{3, 2, 1},
                std::array<Float, 3>{2, 3, -1},
                std::array<Float, 3>{1, 1, 0.5}
            };
            SquareMatrix<3> M3_1{data_1d};
            SquareMatrix<3> M3_2{data};

            debugOutput(M3_1);
            debugOutput(M3_2);
            assert(M3_1[(Vector2<std::size_t> {1, 2})] == Float{6.5});
            assert(M3_2[(Vector2<std::size_t> {1, 2})] == Float{-1});
        }
        std::cout << "3. Operator overloading\n";
        {
            std::array<Float, 9> data_1d = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
            std::array<std::array<Float, 3>, 3> data = {
                std::array<Float, 3>{3, 2, 1},
                std::array<Float, 3>{2, 3, -1},
                std::array<Float, 3>{1, 1, 0.5}
            };
            SquareMatrix<3> M3_1{data_1d};
            SquareMatrix<3> M3_2{data};

            auto M3_3 = M3_1 + M3_2;
            auto M3_4 = M3_1 - M3_2;
            auto M3_5 = M3_1 * Float{2};
            auto M3_6 = Float{2} * M3_1;
            auto M3_7 = M3_1 / Float{2};

            debugOutput(M3_3);
            debugOutput(M3_4);
            debugOutput(M3_5);
            debugOutput(M3_6);
            debugOutput(M3_7);

            assert(M3_3[(Vector2<std::size_t> {2, 1})] == Float{9.5});
            assert(M3_4[(Vector2<std::size_t> {2, 1})] == Float{7.5});
            assert(M3_5[(Vector2<std::size_t> {2, 1})] == Float{17});
            assert(M3_6[(Vector2<std::size_t> {2, 1})] == Float{17});
            assert(M3_7[(Vector2<std::size_t> {2, 1})] == Float{4.25});
            assert(M3_1 != M3_2);
            assert(M3_1 < M3_2);
            assert(!(M3_1 < M3_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            std::array<std::array<Float, 3>, 3> data = {
                std::array<Float, 3>{3, 2, 1},
                std::array<Float, 3>{2, 3, -1},
                std::array<Float, 3>{1, 1, 0.5}
            };
            SquareMatrix<3> M3_1;
            SquareMatrix<3> M3_2 = SquareMatrix<3>::Zeros();
            SquareMatrix<3> M3_3{data};
            Float det1 = M3_1.determinant();
            Float det2 = M3_3.determinant();
            SquareMatrix<3> M3_4 = M3_3.inverse();
            SquareMatrix<3> M3_5 = M3_3.transpose();

            debugOutput(M3_1);
            debugOutput(M3_2);
            debugOutput(M3_3);
            debugOutput(M3_2.isIdentity());
            debugOutput(M3_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M3_4);
            debugOutput(M3_5);

            assert(M3_1 == M3_2);
            assert(not M3_3.isIdentity());
            assert(det1 == Float{0});
            assert(det2 == Float{2.5});
            assert(M3_4[(Vector2<std::size_t> {1, 2})] == Float{2});
            assert(M3_5[(Vector2<std::size_t> {1, 2})] == Float{1});
        }
    }

    std::cout << "********** Square Matrix, N = 4 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::array<Float, 16> data_1d = {};
            SquareMatrix<4> M4_1{data_1d};
            std::cout << "Size of `SquareMatrix<4>` = " << sizeof(M4_1) << std::endl;
            assert(sizeof(M4_1) == 16 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            std::array<Float, 16> data_1d = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5};
            std::array<std::array<Float, 4>, 4> data = {
                std::array<Float, 4>{1, 2, 3, 4},
                std::array<Float, 4>{2, 3, 1, 2},
                std::array<Float, 4>{1, 1, 1, -1},
                std::array<Float, 4>{1, 0, -2, -6}
            };
            SquareMatrix<4> M4_1{data_1d};
            SquareMatrix<4> M4_2{data};

            debugOutput(M4_1);
            debugOutput(M4_2);
            assert(M4_1[(Vector2<std::size_t> {2, 3})] == Float{12.5});
            assert(M4_2[(Vector2<std::size_t> {2, 3})] == Float{-1});
        }
        std::cout << "3. Operator overloading\n";
        {
            std::array<Float, 16> data_1d = {0.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5};
            std::array<std::array<Float, 4>, 4> data = {
                std::array<Float, 4>{1, 2, 3, 4},
                std::array<Float, 4>{2, 3, 1, 2},
                std::array<Float, 4>{1, 1, 1, -1},
                std::array<Float, 4>{1, 0, -2, -6}
            };
            SquareMatrix<4> M4_1{data_1d};
            SquareMatrix<4> M4_2{data};

            auto M4_3 = M4_1 + M4_2;
            auto M4_4 = M4_1 - M4_2;
            auto M4_5 = M4_1 * Float{2};
            auto M4_6 = Float{2} * M4_1;
            auto M4_7 = M4_1 / Float{2};

            debugOutput(M4_3);
            debugOutput(M4_4);
            debugOutput(M4_5);
            debugOutput(M4_6);
            debugOutput(M4_7);

            assert(M4_3[(Vector2<std::size_t> {1, 3})] == Float{10.5});
            assert(M4_4[(Vector2<std::size_t> {1, 3})] == Float{6.5});
            assert(M4_5[(Vector2<std::size_t> {1, 3})] == Float{17});
            assert(M4_6[(Vector2<std::size_t> {1, 3})] == Float{17});
            assert(M4_7[(Vector2<std::size_t> {1, 3})] == Float{4.25});
            assert(M4_1 != M4_2);
            assert(M4_1 < M4_2);
            assert(!(M4_1 < M4_1));
        }
        std::cout << "4. Auxiliary functions\n";
        {
            std::array<std::array<Float, 4>, 4> data = {
                std::array<Float, 4>{1, 2, 3, 4},
                std::array<Float, 4>{2, 3, 1, 2},
                std::array<Float, 4>{1, 1, 1, -1},
                std::array<Float, 4>{1, 0, -2, -6}
            };
            SquareMatrix<4> M4_1;
            SquareMatrix<4> M4_2 = SquareMatrix<4>::Zeros();
            SquareMatrix<4> M4_3{data};
            Float det1 = M4_1.determinant();
            Float det2 = M4_3.determinant();
            SquareMatrix<4> M4_4 = M4_3.inverse();
            SquareMatrix<4> M4_5 = M4_3.transpose();

            debugOutput(M4_1);
            debugOutput(M4_2);
            debugOutput(M4_3);
            debugOutput(M4_2.isIdentity());
            debugOutput(M4_3.isIdentity());
            debugOutput(det1);
            debugOutput(det2);
            debugOutput(M4_4);
            debugOutput(M4_5);

            assert(M4_1 == M4_2);
            assert(not M4_3.isIdentity());
            assert(det1 == Float{0});
            assert(det2 == Float{-1});
            assert(M4_4[(Vector2<std::size_t> {1, 2})] == Float{20});
            assert(M4_5[(Vector2<std::size_t> {1, 2})] == Float{1});
        }
    }

    std::cout << "********** Square Matrix, N > 4 **********\n";
    {
        SquareMatrix<5> M5{
            std::array<Float, 25> {
                2, 1, 3, 7, 5,
                3, 8, 7, 9, 8,
                3, 4, 1, 6, 2,
                4, 0, 2, 2, 3,
                7, 9, 1, 5, 4
            }
        };

        debugOutput(sizeof(SquareMatrix<8>));
        debugOutput(M5.determinant());
        debugOutput(M5);
    }

    return 0;
}
