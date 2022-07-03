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
        }
    }
}
