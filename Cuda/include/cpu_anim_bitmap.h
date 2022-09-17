#ifndef _CPU_ANIM_BITMAP_
#define _CPU_ANIM_BITMAP_

#include <functional>

#include "anim_bitmap.h"

class CPUAnimBitmap: public AnimBitmap {
public:
    std::vector<uint8_t> cpu_bitmap;
    uint8_t* gpu_bitmap;

public:
    CPUAnimBitmap(int width, int height, std::string const& title, void *data);
    virtual ~CPUAnimBitmap();

    void toImage(std::string const& path);
    void animate(std::function<void(uint8_t*, int, int, int)> render_func);
};

#endif // !_CPU_ANIM_BITMAP_
