# Geometry

## Vector

```c++
class Vector2 {
    union {
        T data[2];
        struct { T x, y; };
    };
};
```

这里利用了 `c++` 匿名 `union` 和匿名 `struct` 的特性，使得其内部的成员变量注入到上一层的作用域中。这样我们就可以直接在 `Vector2` 的实例里面直接使用 `data, x, y`。匿名 `union` 详见 [Anonymous Union](https://en.cppreference.com/w/cpp/language/union)。

虽然我们用的是 `struct`，不需要用 `friend` 就可以访问 `Vector` 的成员变量。但是这里我们用 `friend` 来表示这是一个非成员函数。

### Point

先用 Vector 顶替一下。

### Normal

先用 Vector 顶替一下。

## Ray

光线用
$$
\mathbf{r}=\mathbf{o}+t\mathbf{d}
$$
表示。而在 `Ray` 中的 `time` 表示动态场景的时间，和 $t$ 不是一个意思。

### Ray Differentials

先跳过。





