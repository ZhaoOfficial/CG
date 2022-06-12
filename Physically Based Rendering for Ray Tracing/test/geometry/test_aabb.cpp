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
            Bbox2f box3{ mini, maxi };

            auto diag = box2.diagonal();
            auto measure = box2.measure();
            auto norm_coord = box2.normCoord(Point2f{ 2, 3 });
            auto real_coord = box2.realCoord(norm_coord);
            box3.expand(Point2f{ 5, 6 });
            auto box4 = box2.unions(box3);
            bool contain = box4.contains(Point2f{ 3, 3 });
            bool overlap = box4.overlap(box3);
            auto box5 = box4.intersect(box3);
            std::cout << "box1.empty() = " << box1.empty() << "\nbox2.empty() = " << box2.empty() << std::endl;
            std::cout << "diag = " << diag << std::endl;
            std::cout << "measure = " << measure << std::endl;
            std::cout << "norm_coord = " << norm_coord << std::endl;
            std::cout << "real_coord = " << real_coord << std::endl;
            std::cout << "contain = " << contain << std::endl;
            std::cout << "overlap = " << overlap << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
            std::cout << "box4 = " << box4 << std::endl;
            std::cout << "box5 = " << box5 << std::endl;
            std::cout << "corners = " << box2.corner(0) << ' ' << box2.corner(1) << ' ' << box2.corner(2) << ' ' << box2.corner(3) << std::endl;
            assert(box1.empty() == true and box2.empty() == false);
            assert((diag == Vector2f{ 2, 2 }));
            assert(measure == Float(4));
            assert((norm_coord == Vector2f{ 0.5, 0.5 }));
            assert((real_coord == Point2f{ 2, 3 }));
            assert((box3 == Bbox2f{ Point2f{ 1, 2 }, Point2f{ 5, 6 } }));
            assert((box3 == box4));
            assert((box3 == box5));
            assert(contain and overlap);
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
            Bbox3f box3{ mini, maxi };

            auto diag = box2.diagonal();
            auto measure = box2.measure();
            auto norm_coord = box2.normCoord(Point3f{ 2, 3, 4 });
            auto real_coord = box2.realCoord(norm_coord);
            box3.expand(Point3f{ 7, 8, 9 });
            auto box4 = box2.unions(box3);
            bool contain = box4.contains(Point3f{ 3, 3, 3 });
            bool overlap = box4.overlap(box3);
            auto box5 = box4.intersect(box3);
            std::cout << "box1.empty() = " << box1.empty() << "\nbox2.empty() = " << box2.empty() << std::endl;
            std::cout << "diag = " << diag << std::endl;
            std::cout << "measure = " << measure << std::endl;
            std::cout << "norm_coord = " << norm_coord << std::endl;
            std::cout << "real_coord = " << real_coord << std::endl;
            std::cout << "contain = " << contain << std::endl;
            std::cout << "overlap = " << overlap << std::endl;
            std::cout << "box3 = " << box3 << std::endl;
            std::cout << "box4 = " << box4 << std::endl;
            std::cout << "box5 = " << box5 << std::endl;
            std::cout << "corners = " << box2.corner(0) << ' ' << box2.corner(1) << ' ' << box2.corner(2) << ' ' << box2.corner(3) << ' ' << box2.corner(4) << ' ' << box2.corner(5) << ' ' << box2.corner(6) << ' ' << box2.corner(7) << std::endl;
            assert(box1.empty() == true and box2.empty() == false);
            assert((diag == Vector3f{ 3, 3, 3 }));
            assert(measure == Float(27));
            assert((norm_coord == Vector3f{ 1, 1, 1 } / Float(3)));
            assert((real_coord == Point3f{ 2, 3, 4 }));
            assert((box3 == Bbox3f{ Point3f{ 1, 2, 3 }, Point3f{ 7, 8, 9 } }));
            assert((box3 == box4));
            assert((box3 == box5));
            assert(contain and overlap);
        }
    }

    return 0;
}
