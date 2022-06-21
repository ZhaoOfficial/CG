#include <cassert>

#include "debug.h"
#include "math/arithmetic.h"

using namespace pbrt;

int main(int argc, char **argv) {

    std::cout << "********** Floating point **********\n";
    {
        float a = PI;
        float b = -FRAC_1_PI;
        float c = SQRT_2;
        float d = -SQRT_3;

        double e = PI;
        double f = -FRAC_1_PI;
        double g = SQRT_2;
        double h = -SQRT_3;

        std::cout << "1. FloatingPointBitsType\n";
        {
            debugOutput(typeid(FloatingPointBitsType<float>).name());
            debugOutput(typeid(FloatingPointBitsType<double>).name());

            static_assert(FloatingPointType<float>);
            static_assert(FloatingPointType<double>);
            static_assert(not FloatingPointType<long double>);
            static_assert(std::is_same_v<FloatingPointBitsType<float>, uint32_t>);
            static_assert(std::is_same_v<FloatingPointBitsType<double>, uint64_t>);
        }
        std::cout << "2. fromBits\n";
        {
            std::cout << std::hex;
            debugOutput(FloatingPoint::toBits(a));
            debugOutput(FloatingPoint::toBits(b));
            debugOutput(FloatingPoint::toBits(c));
            debugOutput(FloatingPoint::toBits(d));
            debugOutput(FloatingPoint::toBits(e));
            debugOutput(FloatingPoint::toBits(f));
            debugOutput(FloatingPoint::toBits(g));
            debugOutput(FloatingPoint::toBits(h));
            std::cout << std::dec;
        }
        std::cout << "3. getSignBit\n";
        {
            std::cout << std::hex;
            debugOutput(FloatingPoint::getSignBit(a));
            debugOutput(FloatingPoint::getSignBit(b));
            debugOutput(FloatingPoint::getSignBit(c));
            debugOutput(FloatingPoint::getSignBit(d));
            debugOutput(FloatingPoint::getSignBit(e));
            debugOutput(FloatingPoint::getSignBit(f));
            debugOutput(FloatingPoint::getSignBit(g));
            debugOutput(FloatingPoint::getSignBit(h));
            std::cout << std::dec;

            assert(FloatingPoint::getSignBit(a) == (std::bit_cast<uint32_t>(a) & 0x80000000));
            assert(FloatingPoint::getSignBit(b) == (std::bit_cast<uint32_t>(b) & 0x80000000));
            assert(FloatingPoint::getSignBit(c) == (std::bit_cast<uint32_t>(c) & 0x80000000));
            assert(FloatingPoint::getSignBit(d) == (std::bit_cast<uint32_t>(d) & 0x80000000));
            assert(FloatingPoint::getSignBit(e) == (std::bit_cast<uint64_t>(e) & 0x8000000000000000));
            assert(FloatingPoint::getSignBit(f) == (std::bit_cast<uint64_t>(f) & 0x8000000000000000));
            assert(FloatingPoint::getSignBit(g) == (std::bit_cast<uint64_t>(g) & 0x8000000000000000));
            assert(FloatingPoint::getSignBit(h) == (std::bit_cast<uint64_t>(h) & 0x8000000000000000));
        }
        std::cout << "4. getExponent\n";
        {
            std::cout << std::hex;
            debugOutput(FloatingPoint::getExponent(a));
            debugOutput(FloatingPoint::getExponent(b));
            debugOutput(FloatingPoint::getExponent(c));
            debugOutput(FloatingPoint::getExponent(d));
            debugOutput(FloatingPoint::getExponent(e));
            debugOutput(FloatingPoint::getExponent(f));
            debugOutput(FloatingPoint::getExponent(g));
            debugOutput(FloatingPoint::getExponent(h));
            std::cout << std::dec;

            assert(FloatingPoint::getExponent(a) == (std::bit_cast<uint32_t>(a) & 0x7f800000));
            assert(FloatingPoint::getExponent(b) == (std::bit_cast<uint32_t>(b) & 0x7f800000));
            assert(FloatingPoint::getExponent(c) == (std::bit_cast<uint32_t>(c) & 0x7f800000));
            assert(FloatingPoint::getExponent(d) == (std::bit_cast<uint32_t>(d) & 0x7f800000));
            assert(FloatingPoint::getExponent(e) == (std::bit_cast<uint64_t>(e) & 0x7ff0000000000000));
            assert(FloatingPoint::getExponent(f) == (std::bit_cast<uint64_t>(f) & 0x7ff0000000000000));
            assert(FloatingPoint::getExponent(g) == (std::bit_cast<uint64_t>(g) & 0x7ff0000000000000));
            assert(FloatingPoint::getExponent(h) == (std::bit_cast<uint64_t>(h) & 0x7ff0000000000000));
        }
        std::cout << "54.getSignificand\n";
        {
            std::cout << std::hex;
            debugOutput(FloatingPoint::getSignificand(a));
            debugOutput(FloatingPoint::getSignificand(b));
            debugOutput(FloatingPoint::getSignificand(c));
            debugOutput(FloatingPoint::getSignificand(d));
            debugOutput(FloatingPoint::getSignificand(e));
            debugOutput(FloatingPoint::getSignificand(f));
            debugOutput(FloatingPoint::getSignificand(g));
            debugOutput(FloatingPoint::getSignificand(h));
            std::cout << std::dec;

            assert(FloatingPoint::getSignificand(a) == (std::bit_cast<uint32_t>(a) & 0x007fffff));
            assert(FloatingPoint::getSignificand(b) == (std::bit_cast<uint32_t>(b) & 0x007fffff));
            assert(FloatingPoint::getSignificand(c) == (std::bit_cast<uint32_t>(c) & 0x007fffff));
            assert(FloatingPoint::getSignificand(d) == (std::bit_cast<uint32_t>(d) & 0x007fffff));
            assert(FloatingPoint::getSignificand(e) == (std::bit_cast<uint64_t>(e) & 0x000fffffffffffff));
            assert(FloatingPoint::getSignificand(f) == (std::bit_cast<uint64_t>(f) & 0x000fffffffffffff));
            assert(FloatingPoint::getSignificand(g) == (std::bit_cast<uint64_t>(g) & 0x000fffffffffffff));
            assert(FloatingPoint::getSignificand(h) == (std::bit_cast<uint64_t>(h) & 0x000fffffffffffff));
        }
    }

    std::cout << "\n";
    return 0;
}
