# Lecture

## 09 Ray tracing basics

### Ray‐geometry intersection

#### Bounding box

The box with the smallest measure within which the object lies.

Axis‐aligned minimum bounding box (AABB)

#### Ray equation

$$
\mathbf r(t)=\mathbf o+t\mathbf d
$$

#### Ray‐Sphere intersection

设圆方程为
$$
f(\mathbf x)=|\mathbf x|^2-r^2
$$
带入光线求解：
$$
f(\mathbf r(t))=|\mathbf o+t\mathbf d|^2-r^2\\
|\mathbf d|^2t^2+2(\mathbf o\cdot\mathbf d)t+|\mathbf o|^2-r^2=0\\
t=\frac{-(\mathbf o\cdot\mathbf d)\pm\sqrt{(\mathbf o\cdot\mathbf d)^2-|\mathbf d|^2(|\mathbf o|^2-r^2)}}{|\mathbf d|^2}
$$
如果方向是单位向量：
$$
t=-(\mathbf o\cdot\mathbf d)\pm\sqrt{(\mathbf o\cdot\mathbf d)^2-|\mathbf o|^2+r^2}
$$

#### Ray‐plane intersection

设平面方程为
$$
f(\mathbf x)=\mathbf n^T\mathbf x-c=0
$$
带入光线求解：
$$
\mathbf n^T(\mathbf o+t\mathbf d)=c\\
t=\frac{c-\mathbf n^T\mathbf o}{\mathbf n^T\mathbf d}
$$

#### Ray‐bounding‐box intersection

- Insert ray equation into the plane
- Get insertion position, since it is the AABB, the we need to get 2 component of position 
- Check if it is within the AABB range

#### Ray‐triangle intersection

先做光线和平面相交求出相交点。

在三角形重心坐标系中，坐标分量和等于 $1$ 且分量都大约等于 $0$，则代表点在三角形内。因此设交点坐标为：
$$
(1-b_1-b_2)\mathbf p_0+b_1\mathbf p_1+b_2\mathbf p_2\\
b_1,b_2\ge0,b_1+b_2\le1
$$
则有：
$$
\mathbf o+t\mathbf d=(1-b_1-b_2)\mathbf p_0+b_1\mathbf p_1+b_2\mathbf p_2\\
\mathbf p_0-\mathbf o=\mathbf d t+(\mathbf p_0-\mathbf p_1)b_1+(\mathbf p_0-\mathbf p_2)b_2\\
\begin{bmatrix}\mathbf d&\mathbf p_0-\mathbf p_1&\mathbf p_0-\mathbf p_2\end{bmatrix}\begin{bmatrix}t\\b_1\\b_2\end{bmatrix}=\mathbf p_0-\mathbf o
$$
设 $\mathbf a=\mathbf p_0-\mathbf 0, \mathbf b=\mathbf p_0-\mathbf p_1,\mathbf c=\mathbf p_0-\mathbf p_2$。

用 Cramer 公式：
$$
\begin{bmatrix}t\\b_1\\b_2\end{bmatrix}
=
\frac{1}{\begin{vmatrix}\mathbf d&\mathbf b&\mathbf c\end{vmatrix}}
\begin{bmatrix}
\begin{vmatrix}\mathbf a&\mathbf b&\mathbf c\end{vmatrix}\\
\begin{vmatrix}\mathbf d&\mathbf a&\mathbf c\end{vmatrix}\\
\begin{vmatrix}\mathbf d&\mathbf b&\mathbf a\end{vmatrix}\\
\end{bmatrix}
$$
简化：
$$
\begin{align*}
\begin{bmatrix}t\\b_1\\b_2\end{bmatrix}
&=
\frac{1}{\begin{vmatrix}\mathbf d&\mathbf b&\mathbf c\end{vmatrix}}
\begin{bmatrix}
\begin{vmatrix}\mathbf a&\mathbf b&\mathbf c\end{vmatrix}\\
\begin{vmatrix}\mathbf d&\mathbf a&\mathbf c\end{vmatrix}\\
\begin{vmatrix}\mathbf d&\mathbf b&\mathbf a\end{vmatrix}\\
\end{bmatrix}\\
&=\frac{1}{(\mathbf d\times\mathbf b)\cdot\mathbf c}
\begin{bmatrix}
-(\mathbf a\times\mathbf c)\cdot\mathbf b\\
(\mathbf a\times\mathbf c)\cdot\mathbf d\\
(\mathbf d\times\mathbf b)\cdot\mathbf a\\
\end{bmatrix}\\
\end{align*}
$$
然后检查系数即可。交点法向插值即可得到。

#### Ray‐mesh intersection

Grid acceleration structure, each grid record the index of triangle that overlap with. The number of grids is proportion to total number of primitives.

Octree for multi-level grid.

Bounding volume hierarchies for sparse scene.

### Shading

Environment mapping

### More advanced camera models

Aperture

### Anti‐aliasing

Super‐sampling

# Ray Tracing in One Weekend

## Output an Image

用 `.ppm` 格式存储图片. Use `.ppm` format to save the image.

## The `vec3` Class

## Rays, a Simple Camera, and Background

Let ray be a line: $\mathbf p(t)=\mathbf a+t\mathbf b$, where $\mathbf a$ is the ray origin and $\mathbf b$ is the ray direction. Here $t\in(0,\infty)$.

光追: 光线看到的颜色. Ray Tracing: What color is seen along a ray.

1. 计算从相机到像素的光线. Calculate the ray from the camera to the pixel.
2. 决定哪些物体和光线相交. Determine which objects the ray intersects.
3. 计算相交点的颜色. Compute a color for that intersection point.

Set the x-axis go right, the y-axis go up, the z-axis go into screen. Set camera at $(0,0,0)$.

## Adding a Sphere

Let the center of the sphere be $\mathbf c(c_x,c_y,c_z)$, for any point $\mathbf p = (x,y,z)$:
$$
(\mathbf c - \mathbf p)\cdot(\mathbf c - \mathbf p)=(x-c_x)^2+(y-c_y)^2+(z-c_z)^2
$$
is the squared distance between $\mathbf c$ and $\mathbf p$.

For any point $\mathbf p = (x,y,z)$ on sphere:
$$
(\mathbf c - \mathbf p)\cdot(\mathbf c - \mathbf p)=(x-c_x)^2+(y-c_y)^2+(z-c_z)^2=r^2
$$
For any ray $\mathbf p(t)=\mathbf a+t\mathbf b$ hit the sphere:
$$
(\mathbf c - \mathbf p(t))\cdot(\mathbf c - \mathbf p(t))=r^2\\
(\mathbf a+t\mathbf b-\mathbf c)\cdot(\mathbf a+t\mathbf b-\mathbf c)=r^2\\
\underbrace{\mathbf b\cdot\mathbf b}_{a}t^2+\underbrace{2\mathbf b\cdot(\mathbf a-\mathbf c)}_{b}t+\underbrace{(\mathbf a-\mathbf c)\cdot(\mathbf a-\mathbf c)-r^2}_{c}=0
$$
So if this equation has root, then the ray hit the sphere.

## Surface Normals and Multiple Objects

在上面的方程中, 可能会出现两个解, 我们需要的是靠近相机的接触点, 即 $t$ 比较小的解, 因此选取
$$
\frac{-b-\sqrt{b^2-4ac}}{2a}
$$
但可能球把相机包裹住了，这个解 $t<0$, 因此此时我们需要可能符合条件的另外一个解.

观察到 $b=2\mathbf b\cdot(\mathbf a-\mathbf c)=2h$, 带入可得:
$$
\frac{-h-\sqrt{h^2-ac}}{a}
$$
可以减少计算量.

球心到球上一点的向量就是该点的法向量. 
Sphere surface-normal is a vector from the center to a point on sphere.

这个法向量指向球面外侧, 但为了区分内外, 我们也需要内法向. 为方便判断, 我们存储外法向, 通过点乘判断光线从内还是外发射. 

## Antialiasing

真实的相机拍出来的图片,　物体的周围都不是锯齿状的, 都会和旁边的像素进行混合. 
When a real camera takes a picture, there are usually no jaggies along edges because the edge pixels are a blend of some foreground and some background.

每个像素点多采样几次, 采样时候给横向和纵向一个扰动, 使得在边缘时候光线不一定严格落在球的范围内(外), 多次采样取平均即可得到该点像素. 
For a given pixel we have several samples within that pixel and send rays through each of the samples.

## Diffuse Materials

光线可能被反射, 更可能被吸收. 越黑的表面吸收度越高. 
Rays might be absorbed rather than reflected. The darker the surface, the more likely absorption is.

漫反射一个不那么好的模型是随机吸收和随机方向反射光线. 在反射点处画一个与物体外切的单位球, 在单位球中选一个方向就是反射光线的方向.
Really any algorithm that randomizes direction will produce surfaces that look matte.

设光线和球相交于 $\mathbf p$ 点， $\mathbf n$ 为表面外法向，我们把 $\mathbf p+\mathbf n$ 看作球的单位外切球的球心。在这个单位外切球中随机选取一个点 $\mathbf s$，以 $\mathbf s-\mathbf p$ 作为光线的反射方向。这个方法很大概率会选出和 $\mathbf n$ 比较靠近的向量。

漫反射一个比较好的模型是 Lambertian Reflection. 

我们改为在单位外切球面上选取点，就可以满足漫反射所需的 $\cos$ 条件。

## Metal

金属是镜面反射. 

## Dielectrics

透光介质在光射入时候会产生折射光线和反射光线. When a light ray hits them, it splits into a reflected ray and a refracted (transmitted) ray.

Snell's Law
$$
n_1\sin\theta_1=n_2\sin\theta_2
$$
$\theta_1$ 是入射角, $\theta_2$ 是折射角, $n_1,n_2$ 是相对折射率.

将单位入射光线 $\mathbf r$ (朝内)分解为与法向量 $\mathbf n$ (朝外)平行和垂直的向量
$$
\mathbf r=\mathbf r_{\perp}+\mathbf r_{\parallel}
$$
其中
$$
\begin{align*}
\mathbf r_{\perp}&=\mathbf r-\cos\theta_1\mathbf n\\
\mathbf r_{\parallel}&=\sqrt{1-|\mathbf r_{\perp}|^2}\mathbf n
\end{align*}
$$
则单位折射光线 $\mathbf r'$ 可以表示为
$$
\mathbf r'_{\perp}=\frac{n_1}{n_2}(\mathbf r-\cos\theta_1\mathbf n)\quad\cos\theta_1=\mathbf r\cdot\mathbf n\\
\mathbf r'=\mathbf r'_{\perp}+\mathbf r'_{\parallel}
$$

## Positionable Camera

Field of View (*fov*)

We’ll call the position where we place the camera *lookfrom*, and the point we look at *lookat*.

## Defocus Blur

Depth of Field

我们假设所有光线都刚好在相机镜头上汇聚成像了. 但事实上可能由于镜头畸变和景深, 成像可能并不会完全清晰. 
