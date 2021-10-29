#include <iostream>
#include <fstream>
#include <omp.h>
#include <cstring>
#include <chrono>

#include "utility.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include "color.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "material.hpp"

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    if (depth <= 0)
        return color(0.0f, 0.0f, 0.0f);

    if (world.hit(r, 0.001f, infinity, rec)) {
        color attenuation;
        ray scattered;
        if (world.scatter(r, rec, attenuation, scattered)) {
            return attenuation * ray_color(scattered, world, depth - 1);
        }
        return color(0.0f, 0.0f, 0.0f);
    }

    vec3 unit_direction = r.direction().unit_vector();
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5f, 0.5f, 0.5f));
    world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = uniform_float();
            point3 center(a + uniform_float(), 0.2f, b + uniform_float());

            if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = color::random_in_unit_cube(0.0f, 1.0f) * color::random_in_unit_cube(0.0f, 1.0f);
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                } else if (choose_mat < 0.95f) {
                    // metal
                    auto albedo = color::random_in_unit_cube(0.5f, 1.0f);
                    auto fuzz = uniform_float(0.0f, 0.5f);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5f);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                }
            }
        }
    }

    world.add(make_shared<sphere>(point3(0.0f, 1.0f, 0.0f), 1.0f, make_shared<dielectric>(1.5f)));
    world.add(make_shared<sphere>(point3(-4.0f, 1.0f, 0.0f), 1.0, make_shared<lambertian>(color(0.4f, 0.2f, 0.1f))));
    world.add(make_shared<sphere>(point3(4.0f, 1.0f, 0.0f), 1.0f, make_shared<metal>(color(0.7f, 0.6f, 0.5f), 0.0f)));

    return world;
}

int main() {
    //! image setting
    const int image_width = 3840;
    const int image_height = 2160;
    const int samples_per_pixel = 1000;
    const int max_depth = 50;
    const int thread_num = 12;
    omp_set_num_threads(thread_num);
    color image_line[image_width];
    std::ofstream image("test_ppm.ppm");

    //! world
    auto world = random_scene();

    //! camera setting
    point3 lookfrom(13.0f, 2.0f, 3.0f);
    point3 lookat(0.0f, 0.0f, 0.0f);
    vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float dist_to_focus = 10.0f;

    camera scene_camera(lookfrom, lookat, vup, 20, aperture, dist_to_focus);
    
    //! render
    auto start = std::chrono::high_resolution_clock::now();

    image << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    //* (0, 0) is at left down of image
    //* The rows are written out from top to bottom.
    for (int j = image_height - 1; j >= 0; --j) {
        std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        //* The pixels are written out in rows with pixels left to right.
        #pragma omp parallel for
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (static_cast<float>(i) + gaussian_float()) / static_cast<float>(image_width - 1);
                float v = (static_cast<float>(j) + gaussian_float()) / static_cast<float>(image_height - 1);
                ray r = scene_camera.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            pixel_color /= static_cast<float>(samples_per_pixel);
            image_line[i].v[0] = sqrt(pixel_color.v[0]);
            image_line[i].v[1] = sqrt(pixel_color.v[1]);
            image_line[i].v[2] = sqrt(pixel_color.v[2]);
        }
        write_color(image, image_line, image_width);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << duration << " s\nDone.\n";
    return 0;
}