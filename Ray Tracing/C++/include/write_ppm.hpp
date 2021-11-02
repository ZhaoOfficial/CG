#ifndef _COLOR_HPP_
#define _COLOR_HPP_

#include <fstream>

#include "vec3.hpp"

namespace RayTracing {
	inline void write_ppm(std::ofstream& image, std::vector<color>& pixel_color, const int number);
}

#endif // !_COLOR_HPP_
