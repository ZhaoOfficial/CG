#include <cassert>
#include <iostream>

#include "common.h"
#include "geometry/aabb2.h"
#include "geometry/aabb3.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Bbox2 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Bbox2` = " << sizeof(Bbox2f) << std::endl;
            assert(sizeof(Bbox2f) == 2 * sizeof(Vector2f));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Point2f mini{ 1, 2 };
            Point2f maxi{ 3, 4 };
            Bbox2f box1;
            Bbox2f box2{ mini };
            Bbox2f box3{ mini, maxi };
            std::cout << "box1 = " << box1 << std::endl;
            std::cout << "box2 = " << box2 << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
        }
        std::cout << "3. Operator overloading\n";
        {
            Point2f mini{ 1, 2 };
            Point2f maxi{ 3, 4 };
            Bbox2f box1{ mini };
            Bbox2f box2{ mini, mini };
            Bbox2f box3{ mini, maxi };
            std::cout << "box1 = " << box1 << std::endl;
            std::cout << "box2 = " << box2 << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
            assert(box1 == box2);
            assert(box1 != box3);

            for (std::size_t i{}; i < 2; ++i) {
                std::cout << box3[i] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "4. Auxiliary functions\n";
        {
            Point2f mini{ 1, 2 };
            Point2f maxi{ 3, 4 };
            Bbox2f box1{ mini };
            Bbox2f box2{ mini, maxi };

            auto diag = box2.diagonal();
            auto area = box2.area();
            auto norm_coord = box2.normCoord(Point2f{ 2, 3 });
            auto real_coord = box2.realCoord(norm_coord);
            std::cout << "diag = " << diag << std::endl;
            std::cout << "area = " << area << std::endl;
            std::cout << "norm_coord = " << norm_coord << std::endl;
            std::cout << "real_coord = " << real_coord << std::endl;
            assert((diag == Vector2f{ 2, 2 }));
            assert(area == Float(4));
            assert((norm_coord == Vector2f{ 0.5, 0.5 }));
            assert((real_coord == Point2f{ 2, 3 }));
        }
    }

    std::cout << "********** Bbox3 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Bbox2` = " << sizeof(Bbox3f) << std::endl;
            assert(sizeof(Bbox3f) == 2 * sizeof(Vector3f));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Point3f mini{ 1, 2, 3 };
            Point3f maxi{ 4, 5, 6 };
            Bbox3f box1;
            Bbox3f box2{ mini };
            Bbox3f box3{ mini, maxi };
            std::cout << "box1 = " << box1 << std::endl;
            std::cout << "box2 = " << box2 << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
        }
        std::cout << "3. Operator overloading\n";
        {
            Point3f mini{ 1, 2, 3 };
            Point3f maxi{ 4, 5, 6 };
            Bbox3f box1{ mini };
            Bbox3f box2{ mini, mini };
            Bbox3f box3{ mini, maxi };
            std::cout << "box1 = " << box1 << std::endl;
            std::cout << "box2 = " << box2 << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
            assert(box1 == box2);
            assert(box1 != box3);

            for (std::size_t i{}; i < 3; ++i) {
                std::cout << box3[i] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << "4. Auxiliary functions\n";
        {
            Point3f mini{ 1, 2, 3 };
            Point3f maxi{ 4, 5, 6 };
            Bbox3f box1{ mini };
            Bbox3f box2{ mini, maxi };

            auto diag = box2.diagonal();
            auto area = box2.area();
            auto norm_coord = box2.normCoord(Point3f{ 2, 3, 4 });
            auto real_coord = box2.realCoord(norm_coord);
            std::cout << "diag = " << diag << std::endl;
            std::cout << "area = " << area << std::endl;
            std::cout << "norm_coord = " << norm_coord << std::endl;
            std::cout << "real_coord = " << real_coord << std::endl;
            assert((diag == Vector3f{ 3, 3, 3 }));
            assert(area == Float(27));
            assert((norm_coord == Vector3f{ 1, 1, 1 } / Float(3)));
            assert((real_coord == Point3f{ 2, 3, 4 }));
        }
    }

    return 0;
}
