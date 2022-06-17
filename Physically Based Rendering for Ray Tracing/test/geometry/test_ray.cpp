#include <cassert>
#include <iostream>

#include "common.h"
#include "debug.h"
#include "geometry/ray.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "Size of `Ray` = " << sizeof(Ray) << std::endl;
    Ray r;
    debugOutput(r);
    Ray s{Point3f{1.0, 1.0, 1.0}, Vector3f{1.0, 0.0, 0.0}, 100, 5};
    debugOutput(s);

    return 0;
}
