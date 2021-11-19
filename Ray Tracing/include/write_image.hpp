#ifndef _WRITE_IMAGE_HPP_
#define _WRITE_IMAGE_HPP_

#include <vector>

#include "vec3.hpp"
#include "stb_image_write.h"

namespace RayTracing {
	void write_ppm(const std::string& path, std::vector<unsigned char>& pixel_color, const int image_width, const int image_height);

	void write_png(const std::string& path, std::vector<unsigned char>& pixel_color, const int image_width, const int image_height);
}

#endif // !_WRITE_IMAGE_HPP_
