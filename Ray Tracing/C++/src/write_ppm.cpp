#include "write_ppm.hpp"

namespace RayTracing {

    inline void write_ppm(std::ofstream& image, std::vector<color>& pixel_color, const int number) {
        // 0 <= r, g, b <= 255
        for (int i = 0; i < number; ++i) {
            image << static_cast<int>(256.0f * std::clamp(pixel_color[i].x(), 0.0f, 0.9999f)) << ' '
                << static_cast<int>(256.0f * std::clamp(pixel_color[i].y(), 0.0f, 0.9999f)) << ' '
                << static_cast<int>(256.0f * std::clamp(pixel_color[i].z(), 0.0f, 0.9999f)) << '\n';
        }
    }

}
