#include <algorithm>

#include "texture.hpp"

namespace RayTracing {
    SolidColor::SolidColor() {}
    SolidColor::SolidColor(Color c) : color(c) {}
    SolidColor::SolidColor(float r, float g, float b) : color(r, g, b) {}

    Color SolidColor::value(float u, float v, const Point3& p) const {
        return this->color;
    }


    CheckerTexture::CheckerTexture() {}
    CheckerTexture::CheckerTexture(
        std::shared_ptr<Texture> even, std::shared_ptr<Texture> odd
    ) : even(even), odd(odd) {}

    CheckerTexture::CheckerTexture(
        Color c1, Color c2
    ) : even(std::make_shared<SolidColor>(c1)), odd(std::make_shared<SolidColor>(c2)) {}

    Color CheckerTexture::value(float u, float v, const Point3& p) const {
        float sines = sin(10.0f * p[0]) * sin(10.0f * p[1]) * sin(10.0f * p[2]);
        if (sines < 0.0f)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }


    NoiseTexture::NoiseTexture() {}
    NoiseTexture::NoiseTexture(float scale) : scale(scale) {}
    Color NoiseTexture::value(float u, float v, const Point3& p) const {
        // return Color(0.5f, 0.5f, 0.5f) * (1.0f + this->perlin_noise.noise(scale * p));
        // return Color(1.0f, 1.0f, 1.0f) * this->perlin_noise.turbulence(scale * p);
        return Color(0.5f, 0.5f, 0.5f) * (1.0f + std::sin(scale * p[2] + 10.0f * this->perlin_noise.turbulence(p)));
    }


    ImageTexture::ImageTexture() : data(nullptr), image_width(0), image_height(0), bytes_per_row(0) {}
    ImageTexture::ImageTexture(const char* file_name) {
        int components_per_pixel = ImageTexture::BYTE_DEPTH;
        this->data = stbi_load(
            file_name,
            &this->image_width,
            &this->image_height,
            &components_per_pixel, ImageTexture::BYTE_DEPTH
        );
        if (this->data == nullptr) {
            std::cerr << "ERROR: Could not load texture image file '" << file_name << "'.\n";
            this->image_width = 0;
            this->image_height = 0;
        }
        this->bytes_per_row = ImageTexture::BYTE_DEPTH * this->image_width;
    }
    ImageTexture::~ImageTexture() {
        delete this->data;
    }

    Color ImageTexture::value(float u, float v, const Point3& p) const {
        // if no image data
        if (this->data == nullptr)
            return Color(0.0, 1.0, 1.0);

        // clamp input texture coordinate to [0, 1] * [1, 0]
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);

        int i = std::min(static_cast<int>(u * this->image_width), this->image_width - 1);
        int j = std::min(static_cast<int>(v * this->image_height), this->image_height - 1);

        const float color_scale = 1.0f / 255.0f;
        unsigned char* pixel = this->data + j * this->bytes_per_row + i * ImageTexture::BYTE_DEPTH;

        return Color(pixel[0] * color_scale, pixel[1] * color_scale, pixel[2] * color_scale);
    }
}
