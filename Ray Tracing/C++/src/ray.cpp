#include "ray.hpp"

namespace RayTracing {

	ray::ray(const point3& origin, const vec3& direction) : ori(origin), dir(direction) {}

	point3 ray::origin() const { return ori; }
	vec3 ray::direction() const { return dir; }
	point3 ray::at(float t) const { return ori + dir * t; }
}
