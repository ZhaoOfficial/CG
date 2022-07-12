#include "debug.h"
#include "geometry/quaternion.h"

using namespace pbrt;

int main(int argc, char **argv) {
    std::cout << "********** Vector2 **********\n";
    {
        std::cout << "1. Size\n";
        {
            std::cout << "Size of `Vector2f`: " << sizeof(Quaternion) << std::endl;
            assert(sizeof(Quaternion) == 4 * sizeof(Float));
        }
        std::cout << "2. Constructor and destructor\n";
        {
            Quaternion q1;
            // Quaternion q2{};
            debugOutput(q1);
            assert((q1.w == Vector3f{0, 0, 0}));
            assert(q1.u == 1);
        }
        std::cout << "3. Operator overloading\n";
        {}
        std::cout << "4. Auxiliary functions\n";
        {}
        std::cout << "5. External functions\n";
        {}
    }
}
