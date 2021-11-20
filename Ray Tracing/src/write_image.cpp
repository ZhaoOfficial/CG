#include <vector>
#include <fstream>

#include "vec3.hpp"
#include "write_image.hpp"

namespace RayTracing {

    void write_ppm(const std::string& path, std::vector<unsigned char>& pixel_color, const int image_width, const int image_height) {
        std::ofstream image(path);
        image << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        // 0 <= r, g, b <= 255
        for (size_t j = 0; j < image_height; ++j) {
            for (size_t i = 0; i < image_width; ++i) {
                size_t pos = (j * image_width + i) * 3;
                image << static_cast<int>(pixel_color[pos]) << ' '
                    << static_cast<int>(pixel_color[pos + 1]) << ' '
                    << static_cast<int>(pixel_color[pos + 2]) << '\n';
            }
        }
    }

    void write_png(const std::string& path, std::vector<unsigned char>& pixel_color, const int image_width, const int image_height) {
        stbi_write_png(path.c_str(), image_width, image_height, 3, pixel_color.data(), 0);
    }

}
