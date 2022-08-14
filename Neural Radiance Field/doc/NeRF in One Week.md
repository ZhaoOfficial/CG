# NeRF in One Week (Python)

我做 NeRF in One Week 的初衷是想做一个属于自己和实验室的*模块化 NeRF*。为了最好的性能，模块化 NeRF 必须要用 C++/CUDA 编程。不过在动手写 CUDA 之前，我想到可以用之前的 Python/Jupyter 版的 NeRF 作为 CUDA 版的模板。为了更好地理解 NeRF，以及给实验室新人一个比较友善的学习体验，我把我对于 NeRF 代码的理解写成一个教程以供参考。

对于看 NeRF in One Week 的你，这里我们要求你：
1. 有 Python 基础，至少对于接下来出现的代码不会感到疑惑。
2. 有 Pytorch 基础，至少看得懂官网教程的代码。
3. 看过并且理解 NeRF 论文，能把论文和代码对应起来。

## 0 环境配置

```bash
conda create -n nerf python=3.9
conda activate nerf
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

## 1 前瞻

NeRF 本质是 Neural Volume Rendering。对于 Volume Rendering，我们通常用光追的方法渲染，这是一个简单的[光追教程](https://raytracing.github.io/)。对于 NeRF，它可以分为图形学部分和神经网络部分。图形学部分主要包括相机模型，成像模型；而神经网络部分主要包括场景的表达。因此我们可以把[光追教程](https://raytracing.github.io/)中对于场景部分的代码替换为神经网络即可。不过这样还是过于凌乱，在稍加整理之后，我们可以得到以下的代码结构：
![]()

我们会根据以上的代码结构图对于 NeRF in One Week 的代码进行讲解。

## 2 输入输出模块

这是整个工程中最简单的部分，且对于 NeRF 的理解没有帮助，可以选择跳过。写这部分的目的只是为以后的 CUDA 版本明确所需的小模块。

### 2.1 

## 3 数据集模块

数据集存储的主要是送入网络的输入，而送入网络的输入是光线上的采样点，即三维坐标 $\mathbf{x}$ 和二维方向 $\mathbf{d}$ （实际上是归一化的三维向量）。由于 NeRF 也算一种 Image-based Rendering，而且我们需要通过已知的相机参数和图片信息生成光线，因此相机类和图片数据集类是必不可少的两个小模块。

### 3.1 相机

图形学中的相机有两种，分别是透视（perspective）投影相机和正交（orthographic）投影相机。

## 4 网络模块

### 4.1 编码模块

编码器的作用是将低维的坐标输入映射到高维编码空间。由于神经网络倾向于学习低频的信息，编码操作使得网络能够区分空间上相邻的两个点。

#### 4.1.1 Positional Encoder

Positional Encoding 是用一系列的 $\sin$ 和 $\cos$ 函数来编码低维输入，具体的数学表达形式为：
$$
(x,y,z)\mapsto(\\
\sin(2^0x),\sin(2^1x),\dots,\sin(2^Lx),\\
\cos(2^0x),\cos(2^1x),\dots,\cos(2^Lx),\\
\sin(2^0y),\dots,\cos(2^0y),\dots,\\
\sin(2^0z),\dots,\cos(2^0z),\dots\\
)
$$
其中 $(x,y,z)$ 既可以表示坐标，也可以表示方向。在 NeRF 原文中，对于坐标的编码频率为 $L=10$，对于方向的编码频率为 $L=4$。
