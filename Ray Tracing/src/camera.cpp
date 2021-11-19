#include "camera.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    Camera::Camera(
        Point3 look_from, Point3 look_at, Vec3 up, float aspect_ratio,
        float vertical_fov, float aperture, float focus_dist,
        float start_time, float end_time
    ) : start_time(start_time), end_time(end_time) {

        float viewport_height = 2.0f * tan(degrees_to_radians(vertical_fov) / 2);

        // camera coordinate
        camera_front = unit_vector(look_from - look_at);
        camera_right = unit_vector(cross(up, camera_front));
        camera_up = cross(camera_front, camera_right);

        // view port coordinate
        origin = look_from;
        horizontal = focus_dist * viewport_height * aspect_ratio * camera_right;
        vertical = focus_dist * viewport_height * camera_up;
        lower_left_corner = origin - horizontal / 2.0f + vertical / 2.0f - focus_dist * camera_front;
        
        // lens
        lens_radius = aperture / 2.0f;
    }

    Ray Camera::get_ray(float u, float v) const {
        Vec3 rd = lens_radius * random_in_unit_circle();
        Vec3 offset = camera_front * rd.x() + camera_right * rd.y();
        return Ray(
            origin + offset,
            lower_left_corner + u * horizontal - v * vertical - origin - offset,
            uniform_float(start_time, end_time)
        );
    }
}
