# Chapter 2 Geometry

## 2.1 Coordinate System

原点和 3 个线性独立向量称为定义坐标系的*参考系*。
The origin and 3 linear independent vectors are called the *frame* that defines the coordinate system.

要定义一个参考系，我们需要一个点和一组向量，但我们只能有意义地讨论与特定参考系相关的点和向量。因此，在三个维度上，我们需要一个原点为 $(0, 0, 0)$ 和基向量 $(1, 0, 0)$, $(0, 1, 0)$, 和 $(0, 0, 1)$ 的标准参考系。所有其他参考系都将根据这个**规范坐标系**来定义，我们称之为*世界坐标系*。
To define a frame we need a point and a set of vectors, but we can only meaningfully talk about points and vectors with respect to a particular frame. Therefore, in three dimensions we need a standard frame with origin $(0, 0, 0)$ and basis vectors $(1, 0, 0)$, $(0, 1, 0)$, and $(0, 0, 1)$. All other frames will be defined with respect to this **canonical coordinate system**, which we call *world space*.

### 2.1.1 Coordinate System Handedness

`pbrt` uses a left-handed coordinate system.
`pbrt` 使用左手坐标系。

## 2.2 Vectors

```c++
template <typename T>
class Tuple2 {
    union {
        T data[2];
        struct { T x, y; };
    };
};
```

这里利用了 `c++` 匿名 `union` 和匿名 `struct` 的特性，使得其内部的成员变量注入到上一层的作用域中。这样我们就可以直接在 `Tuple2` 的实例里面直接使用 `data, x, y`。匿名 `union` 详见 [Anonymous Union](https://en.cppreference.com/w/cpp/language/union)。

虽然我们用的是 `struct`，不需要用 `friend` 就可以访问 `Vector` 的成员变量。但是这里我们用 `friend` 来表示这是一个非成员函数。

另一种实现方式是用一个模板类，该模板类也使用整数维数进行参数化，并使用包含相同多 `T` 值的数组来表示坐标。虽然这种方法会减少代码总量，但向量的各个组件不能作为 `v.x` 等访问。
An alternate implementation would be to have a single template class that is also parameterized with an integer number of dimensions and to represent the coordinates with an array of that many T values. While this approach would reduce the total amount of code, individual components of the vector couldn’t be accessed as `v.x` and so forth.

### 2.2.4 Coordinate System from a Vector

**先跳过。**

## 2.3 Points

**先用 Vector 顶替一下。**

## 2.4 Normals

**先用 Vector 顶替一下。**

## 2.5 Rays

光线用 $\mathbf{r}=\mathbf{o}+t\mathbf{d}$ 表示。而在 `Ray` 中的 `time` 表示动态场景的时间，和 $t$ 不是一个意思。

### 2.5.1 Ray Differentials

**先跳过。**

## 2.6 Bounding Boxes

以 $[0,1]^2$ 的 2 维的 box 为例，`corner()` 函数从 $0$ 到 $3$ 分别返回的是 $(0,0)$，$(1,0)$，$(0,1)$，$(1,1)$。

## 2.7 Transformations

### 2.7.1 Homogeneous Coordinates

```cpp
template <std::size_t N>
class SquareMatrix;
```

这种写法是 `Non-type template paramter`。从 `C++20` 开始，任何类型都可以作为非类型模板参数。而在 `C++17` 时候，浮点数和 `class` 还不能做非类型模板参数。

`pbrt` 中的 `Shapes` 存储一个指向 `Transform` 的指针。
`Shapes` in `pbrt` store a pointer to a `Transform`.

### 2.7.2 Basic Operations


