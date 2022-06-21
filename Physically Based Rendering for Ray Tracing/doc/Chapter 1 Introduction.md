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



- Light sources
- Visibility
- Surface scattering
- Indirect light transport
- Ray propagation


