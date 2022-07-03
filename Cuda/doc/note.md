# CUDA by Example

## Chapter 1 Why CUDA? Why Now?

## Chapter 2 Getting Started

## Chapter 3 Introduction to CUDA C

### 3.2 A First Program

我们已经看到 CUDA C 需要一种语言方法来将函数标记为设备代码。这没什么特别的。将主机代码发送到一个编译器并将设备代码发送到另一个编译器是一种简写方式。
We have seen that CUDA C needed a linguistic method for marking a function as device code. There is nothing special about this; it is shorthand to send host code to one compiler and device code to another compiler.

尖括号表示我们计划传递给运行时系统的参数。这些不是设备代码的参数，而是影响运行时如何启动我们的设备代码的参数。
The angle brackets denote arguments we plan to pass to the runtime system. These are not arguments to the device code but are parameters that will influence how the runtime will launch our device code.

将参数传递给内核并没有什么特别之处。尽管有尖括号语法，但内核调用的外观和行为与标准 C 中的任何函数调用完全相同。
There is nothing special about passing parameters to a kernel. The angle-bracket syntax notwithstanding, a kernel call looks and acts exactly like any function call in standard C.

这提出了一个微妙但重要的观点。CUDA C 的大部分简单性和强大功能来自于模糊主机和设备代码之间界限的能力。
This raises a subtle but important point. Much of the simplicity and power of CUDA C derives from the ability to blur the line between host and device code.

## Chapter 4 Parallel Programming in CUDA C

### 4.2 CUDA Parallel Programming

但是请注意：您启动的 block 的任何维度都不能超过 65,535。这只是硬件强加的限制，因此如果您尝试使用比这更多的 block 启动，您将开始看到失败。
Be warned, though: No dimension of your launch of blocks may exceed 65,535. This is simply a hardware-imposed limit, so you will start to see failures if you attempt launches with more blocks than this. 

块中的 thread 数不能超过设备属性的 `maxThreadsPerBlock` 字段指定的值。
The number of threads in a block cannot exceed the value specified by the `maxThreadsPerBlock` field of the device properties.

```c
// for GeForce RTX 2060
Maximum number of threads per block: 1024
Maximum sizes of each dimension of a block: 1024 x 1024 x 64
Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
```

## Chapter 5 Thread Cooperation

### 5.3 Shared Memory and Synchronization

CUDA C 编译器处理共享内存中的变量与一般变量不同。它为 GPU 上启动的每个 block 创建变量的副本。该 block 中的每个 thread 共享内存，但 thread 无法看到或修改在其他 block 中看到的此变量的副本。
The CUDA C compiler treats variables in shared memory differently than typical variables. It creates a copy of the variable for each block that you launch on the GPU. Every thread in that block shares the memory, but threads cannot see or modify the copy of this variable that is seen within other blocks.

这提供了一种 block 内的 thread 在计算上进行通信和协作极好的方法。
This provides an excellent means by which threads within a block can communicate and collaborate on computations.

此外，共享内存缓冲区物理上驻留在 GPU 上，而不是驻留在片外 DRAM 中。正因为如此，访问共享内存的延迟往往远低于典型的缓冲区，这使得共享内存可以有效地作为每个块、软件管理的缓存或暂存器。
Furthermore, shared memory buffers reside physically on the GPU as opposed to residing in off-chip DRAM. Because of this, the latency to access shared memory tends to be far lower than typical buffers, making shared memory effective as a per-block, softwaremanaged cache or scratchpad.

由于编译器将为每个 block 创建共享变量的副本，因此我们只需要分配足够的内存，以便块中的每个 thread 都有一个对应位置。
Since the compiler will create a copy of the shared variables for each block, we need to allocate only enough memory such that each thread in the block has an entry.

我们需要一种方法来保证在任何人尝试从该缓冲区读取之前，完成所有这些对共享数组的写入。
We need a method to guarantee that all of these writes to the shared array complete before anyone tries to read from this buffer.

#### 5.3.1 Dot Product

CUDA 架构保证在块中的每个线程都执行了 `__syncthreads()` 之前，没有线程会前进到超出 `__syncthreads()` 的指令。不幸的是，如果 `__syncthreads()` 位于不同的分支中，一些线程将永远无法到达 `__syncthreads()`。因此，由于保证在每个线程执行完之前不能执行 `__syncthreads()` 之后的指令，硬件只是继续等待这些线程，等待，等待...
The CUDA Architecture guarantees that no thread will advance to an instruction beyond the `__syncthreads()` until every thread in the block has executed the `__syncthreads()`. Unfortunately, if the `__syncthreads()` sits in a divergent branch, some of the threads will never reach the `__syncthreads()`. Therefore, because of the guarantee that no instruction after a `__syncthreads()` can be executed before every thread has executed it, the hardware simply continues to wait for these threads. And waits. And waits. Forever.

#### 5.3.2 Shared Memory Bitmap

在任何线程通信之前进行同步。
Synchronization before any thread communication.

## Chapter 6 Constant Memory and Events

### 6.2 Constant Memory

#### 6.2.4 Performance with Constant Memory

将内存声明为 `__constant__` 将我们的使用限制为只读。
Declaring memory as `__constant__` constrains our usage to be read-only.

正如我们之前提到的，与从全局内存中读取相同数据相比，从常量内存中读取可以节省内存带宽。从 64KB 的常量内存中读取可以比全局内存的标准读取节省带宽的原因有两个：
As we previously mentioned, reading from constant memory can conserve memory bandwidth when compared to reading the same data from global memory. There are two reasons why reading from the 64KB of constant memory can save bandwidth over standard reads of global memory:

1. 从常量内存中的单次读取可以广播到其他“附近”线程，有效节省多达 15 次读取。
A single read from constant memory can be broadcast to other "nearby" threads, effectively saving up to 15 reads.
2. 常量内存被缓存，因此连续读取同一地址不会产生任何额外的内存流量。
Constant memory is cached, so consecutive reads of the same address will not incur any additional memory traffic.

在编织的世界中，经线是指被编织成织物的一组线。在 CUDA 架构中，warp 指的是 32 个线程的集合，这些线程“编织在一起”并以同步方式执行。在程序的每一行，warp 中的每个线程对不同的数据执行相同的指令。
In the world of weaving, a warp refers to the group of threads being woven together into fabric. In the CUDA Architecture, a warp refers to a collection of 32 threads that are "woven together" and get executed in lockstep. At every line in your program, each thread in a warp executes the same instruction on different data.

在处理常量内存时，NVIDIA 硬件可以将单个内存读取广播到每个半捆线程。如果半捆中的每个线程都从常量内存中的同一地址请求数据，那么您的 GPU 将只生成一个读取请求，然后将数据广播到每个线程。
When it comes to handling constant memory, NVIDIA hardware can broadcast a single memory read to each half-warp. If every thread in a half-warp requests data from the same address in constant memory, your GPU will generate only a single read request and subsequently broadcast the data to every thread.

在从常量内存中的地址第一次读取之后，其他半捆请求相同地址，因此命中常量缓存，将不会产生额外的内存流量。
After the first read from an address in constant memory, other half-warps requesting the same address, and therefore hitting the constant cache, will generate no additional memory traffic.

虽然当所有 16 个线程都读取相同的地址时它可以显着提高性能，但当所有 16 个线程读取不同的地址时，它实际上会降低性能。
Although it can dramatically accelerate performance when all 16 threads are reading the same address, it actually slows performance to a crawl when all 16 threads read different addresses.

### 6.3 Measuring Performance with Events

使用事件最棘手的部分是由于我们在 CUDA C 中进行的一些调用实际上是异步的。
The trickiest part of using events arises as a consequence of the fact that some of the calls we make in CUDA C are actually asynchronous.

当 `cudaEventSynchronize()` 的调用返回时，我们知道在 `stop` 事件之前所有 GPU 工作都完成了，因此可以安全地读取 `stop` 中记录的时间戳。
When the call to `cudaEventSynchronize()` returns, we know that all GPU work before the `stop` event has completed, so it is safe to read the time stamp recorded in `stop`.

It is worth noting that because CUDA events get implemented directly on the GPU, they are unsuitable for timing mixtures of device and host code. That is, you will get unreliable results if you attempt to use CUDA events to time more than kernel executions and memory copies involving the device.

## Chapter 7 Texture Memory

### 7.2 Texture Memory Overview

纹理缓存专为内存访问模式表现出大量空间局部性的图形应用程序而设计。在计算应用程序中，这大致意味着一个线程可能从一个地址“接近”附近线程读取的地址读取。
Texture caches are designed for graphics applications where memory access patterns exhibit a great deal of spatial locality. In a computing application, this roughly implies that a thread is likely to read from an address “near” the address that nearby threads read.

### 7.3 Simulating Heat Transfer

#### 7.3.4 Using Texture Memory

**Deprecated** 尽管它看起来像一个函数，但 `tex1Dfetch()` 是一个编译器内在函数。而且由于纹理引用必须在文件范围内全局声明，我们不能再将输入和输出缓冲区作为参数传递给 `blend_kernel()`，因为编译器需要在编译时知道哪些纹理应该用 `tex1Dfetch()` 采样。
**Deprecated** Although it looks like a function, `tex1Dfetch()` is a compiler intrinsic. And since texture references must be declared globally at file scope, we can no longer pass the input and output buffers as parameters to `blend_kernel()` because the compiler needs to know at compile time which textures `tex1Dfetch()` should be sampling.
