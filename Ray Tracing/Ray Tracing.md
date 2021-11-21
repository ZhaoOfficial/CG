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
带入光线求解: 
$$
f(\mathbf r(t))=|\mathbf o+t\mathbf d|^2-r^2\\
|\mathbf d|^2t^2+2(\mathbf o\cdot\mathbf d)t+|\mathbf o|^2-r^2=0\\
t=\frac{-(\mathbf o\cdot\mathbf d)\pm\sqrt{(\mathbf o\cdot\mathbf d)^2-|\mathbf d|^2(|\mathbf o|^2-r^2)}}{|\mathbf d|^2}
$$
如果方向是单位向量: 
$$
t=-(\mathbf o\cdot\mathbf d)\pm\sqrt{(\mathbf o\cdot\mathbf d)^2-|\mathbf o|^2+r^2}
$$

#### Ray‐plane intersection

设平面方程为
$$
f(\mathbf x)=\mathbf n^T\mathbf x-c=0
$$
带入光线求解: 
$$
\mathbf n^T(\mathbf o+t\mathbf d)=c\\
t=\frac{c-\mathbf n^T\mathbf o}{\mathbf n^T\mathbf d}
$$

#### Ray‐bounding‐box intersection

- Insert ray equation into the plane
- Get insertion position, since it is the AABB, the we need to get 2 component of position 
- Check if it is within the AABB range

#### Ray‐triangle intersection

先做光线和平面相交求出相交点. 

在三角形重心坐标系中, 坐标分量和等于 $1$ 且分量都大约等于 $0$, 则代表点在三角形内. 因此设交点坐标为: 
$$
(1-b_1-b_2)\mathbf p_0+b_1\mathbf p_1+b_2\mathbf p_2\\
b_1,b_2\ge0,b_1+b_2\le1
$$
则有: 
$$
\mathbf o+t\mathbf d=(1-b_1-b_2)\mathbf p_0+b_1\mathbf p_1+b_2\mathbf p_2\\
\mathbf p_0-\mathbf o=\mathbf d t+(\mathbf p_0-\mathbf p_1)b_1+(\mathbf p_0-\mathbf p_2)b_2\\
\begin{bmatrix}\mathbf d&\mathbf p_0-\mathbf p_1&\mathbf p_0-\mathbf p_2\end{bmatrix}\begin{bmatrix}t\\b_1\\b_2\end{bmatrix}=\mathbf p_0-\mathbf o
$$
设 $\mathbf a=\mathbf p_0-\mathbf 0, \mathbf b=\mathbf p_0-\mathbf p_1,\mathbf c=\mathbf p_0-\mathbf p_2$. 

用 Cramer 公式: 
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
简化: 
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
然后检查系数即可. 交点法向插值即可得到. 

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

# Ray Tracing: In One Weekend

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
但可能球把相机包裹住了, 这个解 $t<0$, 因此此时我们需要可能符合条件的另外一个解.

观察到 $b=2\mathbf b\cdot(\mathbf a-\mathbf c)=2h$, 带入可得:
$$
\frac{-h-\sqrt{h^2-ac}}{a}
$$
可以减少计算量.

球心到球上一点的向量就是该点的法向量. 
Sphere surface-normal is a vector from the center to a point on sphere.

这个法向量指向球面外侧, 但为了区分内外, 我们也需要内法向. 为方便判断, 我们存储外法向, 通过点乘判断光线从内还是外发射. 

一条光线可能和很多的东西都会相交, 因此我们选择最近的那个物体.

## Anti-aliasing

真实的相机拍出来的图片, 物体的周围都不是锯齿状的, 都会和旁边的像素进行混合. 
When a real camera takes a picture, there are usually no jaggies along edges because the edge pixels are a blend of some foreground and some background.

每个像素点多采样几次, 采样时候给横向和纵向一个扰动, 使得在边缘时候光线不一定严格落在球的范围内(外), 多次采样取平均即可得到该点像素. 
For a given pixel we have several samples within that pixel and send rays through each of the samples.

## Diffuse Materials

光线可能被反射, 更可能被吸收. 越黑的表面吸收度越高. 
Rays might be absorbed rather than reflected. The darker the surface, the more likely absorption is.

漫反射一个不那么好的模型是随机吸收和随机方向反射光线. 在反射点处画一个与物体外切的单位球, 在单位球中选一个方向就是反射光线的方向.
Really any algorithm that randomizes direction will produce surfaces that look matte.

设光线和球相交于 $\mathbf p$ 点,  $\mathbf n$ 为表面外法向, 我们把 $\mathbf p+\mathbf n$ 看作球的单位外切球的球心. 在这个单位外切球中随机选取一个点 $\mathbf s$, 以 $\mathbf s-\mathbf p$ 作为光线的反射方向. 这个方法很大概率会选出和 $\mathbf n$ 比较靠近的向量. 

漫反射一个比较好的模型是 Lambertian Reflection. 

我们改为在单位外切球面上选取点, 就可以满足漫反射所需的 $\cos$ 条件.

## Metal

对于漫反射材料, 它即可以看作一直散射并且每次衰减反射率的亮度, 也可以看作不衰减但是每次都吸收部分的光, 也可以将二者结合.
For the Lambertian (diffuse) case we already have, it can either scatter always and attenuate by its reflectance $R$, or it can scatter with no attenuation but absorb the fraction $1−R$ of the rays, or it could be a mixture of those strategies.

金属是镜面反射. 对于镜面反射, 如果法向是 $\mathbf n$, 入射光线是 $\mathbf v$, 则反射光线是:
$$
\mathbf v_{\parallel}-\mathbf v_{\perp}=\mathbf v-2\mathbf v_{\perp}=\mathbf v-2(\mathbf v\cdot\mathbf n)\mathbf n
$$
对于反射, 我们可以使得反射光线产生微小偏移, 即在出射方向增加一个微小的偏移. 

## Dielectrics

电介质在光射入时候会产生折射光线和反射光线.
When a light ray hits them, it splits into a reflected ray and a refracted (transmitted) ray.

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
\mathbf r_{\perp}&=\mathbf r-\cos\theta_1\mathbf n
\end{align*}
$$
折射光线的垂直分量和入射光线的垂直分量满足 Snell‘s law. 则单位折射光线 $\mathbf r'$ 可以表示为:
$$
\mathbf r'_{\perp}=\frac{n_1}{n_2}(\mathbf r-\cos\theta_1\mathbf n)\quad\cos\theta_1=\mathbf r\cdot\mathbf n\\
\mathbf r'_{\parallel}=-(\sqrt{1-\mathbf r'^2_{\perp}})\mathbf n\\
\mathbf r'=\mathbf r'_{\perp}+\mathbf r'_{\parallel}
$$

光线在进入电介质时候和出电介质时候的折射率是成倒数的.

考虑全反射现象, 即光密介质到光疏介质时, 折射率和入射角的正弦值的积大于 1 时候就会发生. 

考虑极化现象, 即电解质大角度观察镜面反射分量会增加.

## Positionable Camera

视野角是相机上下可以观察的角度. 长焦视野角小.
Field of View (*fov*). This is the angle you see through the portal.

相机从所在位置看向目标位置, 所成的向量就是前方向. 和给定的上方向得到右方向, 再得到相机的上方向.

## Depth of Field

我们在真实相机中散焦模糊的原因是因为它们需要一个大洞 (而不仅仅是一个针孔) 来收集光线. 这会使所有东西散焦, 但是如果我们在孔中插入一个镜头, 那么所有东西都会在一定距离处聚焦.
The reason we defocus blur in real cameras is because they need a big hole (rather than just a pinhole) to gather light. This would defocus everything, but if we stick a lens in the hole, there will be a certain distance where everything is in focus.

图形学里面, 我们假设所有光线都刚好在相机镜头上汇聚成像了. 但事实上可能由于镜头畸变和景深, 成像可能并不会完全清晰. 

通常, 所有场景光线都来自观察点. 为了实现散焦模糊, 生成源自以观察点为中心的圆盘内部的随机场景光线. 半径越大, 散焦模糊越大. 您可以将我们的原始相机视为具有半径为零的散焦盘 (根本没有模糊), 因此所有光线都起源于盘中心.
Normally, all scene rays originate from the lookfrom point. In order to accomplish defocus blur, generate random scene rays originating from inside a disk centered at the lookfrom point. The larger the radius, the greater the defocus blur. You can think of our original camera as having a defocus disk of radius zero (no blur at all), so all rays originated at the disk center (lookfrom).

# Ray Tracing: The Next Week

## Motion Blur

现实里面, 快门打开并且停留一段时间. 有可能这段时间里面物体或相机发生移动. 因此相机记录的就是这段时间内像素的平均值.
In a real camera, the shutter opens and stays open for a time interval, and the camera and objects may move during that time. Its really an average of what the camera sees over that interval that we want.

让光线存储其存在的时间.
For this we will first need to have a ray store the time it exists at.

## Bounding Volume Hierarchies

光线和物体相交是光追的瓶颈, 时间复杂度为 $O(n)$. 用 Bounding Volume Hierarchies 可以做到 $O(\log n)$.
The ray-object intersection is the main time-bottleneck in a ray tracer, and the time is linear with the number of objects.

两种常用的物体搜寻是分割空间或分割物体. 
The two most common families of sorting are to 1) divide the space, and 2) divide the objects. 

### Axis-Aligned Bounding Boxes

设光线的方程是 $\mathbf r(t)=\mathbf o+t\mathbf d$, 平面的方程为 $x=x_0$ 和 $x=x_1$, 则交点参数为: 
$$
t_0=\frac{x_0-\mathbf o_x}{\mathbf d_x}\\
t_1=\frac{x_1-\mathbf o_x}{\mathbf d_x}
$$
$y,z$ 方向同理.

如果得到的 $(t_{x0}, t_{x1}),(t_{y0}, t_{y1}), (t_{z0}, t_{z1})$ 有重叠部分, 则说明与 AABB 相交了. 但是很有可能光线方向是负数, 元组中的值最好是前者小后者大; 光线方向可能是平行于某个轴的, 可能产生无穷大. 
First, suppose the ray is traveling in the negative x direction. Second, the divide in there could give us infinities. And if the ray origin is on one of the slab boundaries, we can get a `NaN`. 

`BVH` 对于 `AABB` 就像 `HittableList` 对于 `Hittable` 一样.

递归的把 `BVH` 内部的物体分成两部分即可. 

## Solid Textures

### Texture Coordinates for Spheres

需要把 $\theta,\phi\in[0,\pi]\times[0,2\pi]$ 映射到 $u,v\in[0,1]^2$, 因此
$$
u=\frac{\phi}{2\pi}\\
v=\frac{\theta}{\pi}
$$
通过光线和球相交的单位法向量我们可以得到交点在单位球上的位置.

因此对应关系就是: 
$$
x=\sin\theta\cos\phi\\
y=\cos\theta\\
z=-\sin\theta\sin\phi
$$
即:
$$
\theta=\arccos y\\
\phi=\arctan(-\frac{z}{x})
$$

## Perlin Noise

Perlin 噪声的一个关键部分是它是可重复的: 它以 3D 点作为输入并始终返回相同的随机数. 相邻的点返回相似的数字. Perlin 噪声的另一个重要部分是它简单且快速, 因此它通常作为 hack 来完成.
A key part of Perlin noise is that it is repeatable: it takes a 3D point as input and always returns the same randomish number. Nearby points return similar numbers. Another important part of Perlin noise is that it be simple and fast, so it’s usually done as a hack.

## Image Texture Mapping

在图像中缩放的 $(u,v)$ 的直接方法是将 $u$ 和 $v$ 舍入为整数，并将其用作 $(i,j)$ 像素. 这很尴尬，因为我们不想在更改图像分辨率时必须更改代码. 因此，图形中最通用的非官方标准之一是使用纹理坐标而不是图像像素坐标. 这些只是图像中分数位置的某种形式. 
A direct way to use scaled $(u,v)$ in an image is to round the $u$ and $v$ to integers, and use that as $(i,j)$ pixels. This is awkward, because we don’t want to have to change the code when we change image resolution. So instead, one of the the most universal unofficial standards in graphics is to use texture coordinates instead of image pixel coordinates. These are just some form of fractional position in the image. 

## Rectangles and Lights

### Creating Rectangle Objects

一个平行于坐标轴的矩形可以用 4 个平面围 1 个平面得到.
An axis-aligned rectangle is defined by the lines $x=x_0, x=x_1, y=y_0, y=y_1,z=k$.

先得到:
$$
t=\frac{k-\mathbf o_z}{\mathbf d_z}
$$
然后带入得到交点的 $x,y$ 值:
$$
x=\mathbf o_x+t\mathbf d_x\\
y=\mathbf o_y+t\mathbf d_y\\
$$
然后检查范围即可.

平面的 `AABB` 在某一方向的交点可能会变成无穷远. 为了避免这种状况, 我们给平面在这个方向稍微增厚度.

## Instances

用多个平行于坐标轴的长方形拼成一个盒子。


一个实例是一个几何图元，它以某种方式被移动或旋转。这在光线追踪中特别容易，因为我们不移动任何东西；相反，我们将光线向相反的方向移动。
An instance is a geometric primitive that has been moved or rotated somehow. This is especially easy in ray tracing because we don’t move anything; instead we move the rays in the opposite direction.

旋转的顺序很重要。

## Volumes

穿过密度恒定的体积的射线可以在体积内散射，也可以笔直穿过。当光线穿过体积时，它可能会在任何一点散射。体积越密，这种可能性就越大。光线在任何小距离 $\Delta L$ 内散射的概率为：
A ray going through a volume of constant density can either scatter inside the volume, or it can make it all the way through. As the ray passes through the volume, it may scatter at any point. The denser the volume, the more likely that is. The probability that the ray scatters in any small distance $\Delta L$ is:
$$
p=c\Delta L
$$

# Ray Tracing: The Rest of Your Life

## A Simple Monte Carlo Program

### Stratified Samples

一种先分块然后再采样的方法（NeRF），精度会高很多，收敛也很快。

## One Dimensional MC Integration

要积 $[a,b]$ 上的函数，先选一个 $[a,b]$ 上的概率密度函数，对这个概率密度函数采样，然后把 $f(x)/p(x)$ 求和求平均即可。

## MC Integration on the Sphere of Directions

## Light Scattering

### Albedo

Probability of light scattering: $A$

Probability of light being absorbed: $1−𝐴$

Here $A$ stands for *albedo* (latin for *whiteness*).

## Importance Sampling Materials

Reduce noises in Cornell Box.











