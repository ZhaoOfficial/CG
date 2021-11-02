#ifndef _WRITE_PPM_HPP_
#define _WRITE_PPM_HPP_

#include <fstream>
#include <vector>

#include "vec3.hpp"

namespace RayTracing {
	void write_ppm(std::ofstream& image, std::vector<color>& pixel_color, const int number);
}

#endif // !_WRITE_PPM_HPP_
