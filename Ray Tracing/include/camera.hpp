#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include "ray.hpp"
#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    class Camera {
    public:
        Camera(
            Point3 look_from, Point3 look_at, Vec3 up, float aspect_ratio,
            float vertical_fov, float aperture, float focus_dist,
            float start_time, float end_time
        );

        Ray get_ray(float u, float v) const;

    public:
        // view port coordinate
        Point3 origin;
        Point3 lower_left_corner;
        Vec3 horizontal;
        Vec3 vertical;
        // camera coordinate
        Point3 camera_front;
        Point3 camera_right;
        Point3 camera_up;
        // lens
        float lens_radius;
        // shutter open / close time
        float start_time;
        float end_time;
    };

}

#endif // !_CAMERA_HPP_
