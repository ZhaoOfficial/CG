#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include "utility.hpp"
#include "vec3.hpp"

class camera {
public:
    camera(point3 look_from, point3 look_at, vec3 up, float vertical_fov, float aperture, float focus_dist) {
        float theta = degrees_to_radians(vertical_fov);
        float h = tan(theta / 2);

        float viewport_width = 4.8f * h;
        float viewport_height = 2.7f * h;
        float focal_length = 1.0f;

        camera_front = (look_from - look_at).unit_vector();
        camera_right = up.cross(camera_front);
        camera_up = camera_front.cross(camera_right);

        origin = look_from;
        horizontal = focus_dist * viewport_width * camera_right;
        vertical = focus_dist * viewport_height * camera_up;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * camera_front;
        lens_radius = aperture / 2.0f;
    }

    ray get_ray(float u, float v) const {
        vec3 rd = lens_radius * vec3::random_in_unit_circle();
        vec3 offset = camera_front * rd.x() + camera_right * rd.y();
        return ray(
            origin + offset,
            lower_left_corner + u * horizontal + v * vertical - origin - offset
        );
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    point3 camera_front;
    point3 camera_right;
    point3 camera_up;
    float lens_radius;
};

#endif // !_CAMERA_HPP_
