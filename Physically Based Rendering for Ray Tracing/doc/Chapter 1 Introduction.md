# Chapter 1 Introduction

## 1.1 Literate Programming

## 1.2 Photorealistic Rendering and the Ray-Tracing Algorithm

逼真渲染的目标是创建与同一场景的照片无法区分的 3D 场景图像。
The goal of photorealistic rendering is to create an image of a 3D scene that is indistinguishable from a photograph of the same scene.

光线追踪系统至少模拟以下对象和现象：
 Ray tracing systems simulate at least the following objects and phenomena:

### 1.2.1 Cameras

针孔相机模型。当胶片（或图像）平面位于针孔前面时，针孔通常被称为*眼睛*。
Pin-hold camera model. When the film (or image) plane is in front of the pinhole, the pinhole is frequently referred to as the *eye*.

相机模拟器的一项重要任务是在图像上取一个点并生成*光线*，入射光将对图像这一个位置做出贡献。
An important task of the camera simulator is to take a point on the image and generate *rays* along which incident light will contribute to that image location.

### 1.2.2 Ray–Object Intersections

$$
G(t)=F(x(t),y(t),z(t))=0
$$

除了 $t$ 之外的所有值都是已知的，这为我们提供了一个易于求解 $t$ 的方程。如果没有真正的根，光线就会错过物体；如果有根，最小的正数 $t$ 给出了光线和物体的交点。
All of the values besides $t$ are known, giving us an easily solved equation in $t$. If there are no real roots, the ray misses the object; if there are roots, the smallest positive one gives the intersection point.

尽管许多光线追踪器仅使用法线，但更复杂的渲染系统需要更多信息，例如位置和表面法线相对于表面局部参数化的各种偏导数。
Although many ray tracers operate with only normal, more sophisticated rendering systems require even more information, such as various partial derivatives of position and surface normal with respect to the local parameterization of the surface.

### 1.2.3 Light sources


光线-物体相交阶段为我们提供了一个要着色的点以及该点的局部几何信息。
The ray–object intersection stage gives us a point to be shaded and some information about the local geometry at that point.

### 1.2.4 Visibility

仅当从点到灯光位置的路径畅通无阻时，每个灯光才会为要着色的点提供照明。
Each light contributes illumination to the point being shaded only if the path from the point to the light’s position is unobstructed.

我们只需构造一条新光线，其原点位于表面点，其方向指向光线。这些特殊的光线称为*阴影光线*。
We simply construct a new ray whose origin is at the surface point and whose direction points toward the light. These special rays are called *shadow rays*.

### 1.2.5 Surface scattering



- Indirect light transport
- Ray propagation

