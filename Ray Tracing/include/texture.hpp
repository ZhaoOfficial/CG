#ifndef _TEXTURE_HPP_
#define _TEXTURE_HPP_

#include <memory>

#include "stb_image.h"

#include "perlin.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class Texture {
    public:
        virtual Color value(float u, float v, const Point3& p) const = 0;
    };

    class SolidColor : public Texture {
    public:
        SolidColor();
        SolidColor(Color c);
        SolidColor(float r, float g, float b);

        virtual Color value(float u, float v, const Point3& p) const override;

    private:
        Color color;
    };

    class CheckerTexture : public Texture {
    public:
        CheckerTexture();
        CheckerTexture(std::shared_ptr<Texture> even, std::shared_ptr<Texture> odd);
        CheckerTexture(Color c1, Color c2);

        virtual Color value(float u, float v, const Point3& p) const override;

    public:
        std::shared_ptr<Texture> even;
        std::shared_ptr<Texture> odd;
    };

    class NoiseTexture : public Texture {
    public:
        NoiseTexture();
        NoiseTexture(float scale);

        virtual Color value(float u, float v, const Point3& p) const override;
    
    private:
        Perlin perlin_noise;
        float scale;
    };

    class ImageTexture : public Texture {
    public:
        static const int BYTE_DEPTH = 3;

        ImageTexture();
        ImageTexture(const char* file_name);
        ~ImageTexture();

        virtual Color value(float u, float v, const Point3& p) const override;

    private:
        unsigned char *data;
        int image_width;
        int image_height;
        int bytes_per_row;
    };

}

#endif // !_TEXTURE_HPP_
