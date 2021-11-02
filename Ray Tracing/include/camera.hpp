#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include "ray.hpp"
#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    class camera {
    public:
        camera(point3 look_from, point3 look_at, vec3 up, float vertical_fov, float aperture, float focus_dist);

        ray get_ray(float u, float v) const;

    public:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        point3 camera_front;
        point3 camera_right;
        point3 camera_up;
        float lens_radius;
    };

}

#endif // !_CAMERA_HPP_
