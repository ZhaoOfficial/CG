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

## Chapter 8 Graphics Interoperability

### 8.2 Graphics Interoperation

We can use the GPU for both rendering and general-purpose computation.

**Deprecated** 教程里面会让你选择一个可以跑 CUDA 的 GPU 和 OpenGL 绑定。但现在不需要这一步了。

共享数据缓冲区是 CUDA C 内核和 OpenGL 渲染之间互操作的关键组件。要在 OpenGL 和 CUDA 之间传递数据，我们首先需要创建一个可以与这两种 API 一起使用的缓冲区。
Shared data buffers are the key component to interoperation between CUDA C  kernels and OpenGL rendering. To pass data between OpenGL and CUDA, we will  first need to create a buffer that can be used with both APIs.

### 8.4 Heat Transfer with Graphics Interoperability

`CPUAnimBitmap` 的 `Draw()` 例程中的此调用会触发将 `bitmap->pixels` 中的 CPU 缓冲区复制到 GPU 以进行渲染。为此，CPU 需要停止它正在做的事情，并为每一帧启动一份到 GPU 上的副本。这需要 CPU 和 GPU 之间的同步以及额外的延迟来启动和完成通过 PCI Express 总线的传输。由于对 `glDrawPixels()` 的调用在最后一个参数中需要一个 CPU 指针，这也意味着在使用 CUDA C 内核生成一帧图像数据后，我们的第5章 ripple 程序需要将帧从 GPU 用 `cudaMemcpy()` 复制到 CPU。
This call in the `Draw()` routine of `CPUAnimBitmap` triggers a copy of the CPU buffer in `bitmap->pixels` to the GPU for rendering. To do this, the CPU needs to stop what it’s doing and initiate a copy onto the GPU for every frame. This requires synchronization between the CPU and GPU and additional latency to initiate and complete a transfer over the PCI Express bus. Since the call to `glDrawPixels()` expects a host pointer in the last argument, this also means that after generating a frame of image data with a CUDA C kernel, our Chapter 5 ripple application needed to copy the frame from the GPU to the CPU with a `cudaMemcpy()`.

我们使用 CUDA C 计算每帧渲染的图像值，但在计算完成后，我们将缓冲区复制到 CPU，然后 CPU 将缓冲区复制回 GPU 进行显示。这意味着我们在 CPU 和 GPU 之间引入了不必要的数据传输，从而影响了最佳性能。
We used CUDA C to compute image values for our rendering in each frame, but after the computations were done, we copied the buffer to the CPU, which then copied the buffer back to the GPU for display. This means that we introduced unnecessary data transfers between the host and the device that stood between us and maximum performance.

## Chapter 9 Atomic

### 9.2 Compute Capability

#### 9.2.1 The Compute Capability of NVIDIA GPUs

例如，基于 CUDA 架构构建的每个 GPU 都可以启动内核、访问全局内存以及从常量和纹理内存中读取数据。但就像不同型号的 CPU 具有不同的功能和指令集（例如，MMX、SSE 或 SSE2）一样，支持 CUDA 的图形处理器也是如此。NVIDIA 将 GPU 支持的功能称为其计算能力。
For example, every GPU built on the CUDA Architecture can launch kernels, access global memory, and read from constant and texture memories. But just like different models of CPUs have varying capabilities and instruction sets (for example, MMX, SSE, or SSE2), so too do CUDAenabled graphics processors. NVIDIA refers to the supported features of a GPU as its compute capability.

#### 9.2.2 Compiling for A Minimum Compute Capability

```bash
nvcc -arch=sm_xy
```

它通知编译器代码需要 x.y 或更高的计算能力。
It inform the compiler that the code requires compute capability x.y or greater.

## 10 Streams

### 10.2 Paged-Locked Host Memory

C 库函数 `malloc()` 分配标准的可分页主机内存，而 `cudaHostAlloc()` 分配页面锁定主机内存的缓冲区。有时称为**固定**内存，页面锁定缓冲区有一个重要属性：操作系统向我们保证它永远不会将该内存分页到磁盘，从而确保其驻留在物理内存中。这样做的必然结果是操作系统允许应用程序访问内存的物理地址变得安全，因为缓冲区不会被驱逐或重定位。
The C library function `malloc()` allocates standard, pageable host memory, while `cudaHostAlloc()` allocates a buffer of page-locked host memory. Sometimes called **pinned** memory, page-locked buffers have an important property: The operating system guarantees us that it will never page this memory out to disk, which ensures its residency in physical memory. The corollary to this is that it becomes safe for the OS to allow an application access to the physical address of the memory, since the buffer will not be evicted or relocated.

知道缓冲区的物理地址后，GPU 可以使用直接内存访问 (DMA) 将数据复制到主机或从主机复制数据。由于 DMA 复制在没有 CPU 干预的情况下进行，这也意味着 CPU 可以同时将这些缓冲区分页到磁盘或通过更新操作系统的页表来重新定位它们的物理地址。CPU 移动可分页数据的可能性意味着使用固定内存进行 DMA 复制是必不可少的。事实上，即使您尝试使用可分页内存执行内存复制，CUDA 驱动程序仍然使用 DMA 将缓冲区传输到 GPU。因此，您的复制发生了两次，首先从可分页系统缓冲区到页面锁定的“暂存”缓冲区，然后从页面锁定的系统缓冲区到 GPU。
Knowing the physical address of a buffer, the GPU can then use direct memory access (DMA) to copy data to or from the host. Since DMA copies proceed without intervention from the CPU, it also means that the CPU could be simultaneously paging these buffers out to disk or relocating their physical address by updating the operating system’s pagetables. The possibility of the CPU moving pageable data means that using pinned memory for a DMA copy is essential. In fact, even when you attempt to perform a memory copy with pageable memory, the CUDA driver still uses DMA to transfer the buffer to the GPU. Therefore, your copy happens twice, first from a pageable system buffer to a page-locked "staging" buffer and then from the page-locked system buffer to the GPU.

因此，无论何时从可分页内存执行内存复制，您都可以保证复制速度将受 PCIE 传输速度和系统前端总线速度中较低者的限制。在某些系统中，这些总线之间的巨大带宽差异确保了当用于在 GPU 和主机之间复制数据时，页面锁定的主机内存比标准可分页内存具有大约两倍的性能优势。但即使在 PCI Express 和前端总线速度相同的世界中，可分页缓冲区仍会产生额外的 CPU 管理副本的开销。
As a result, whenever you perform memory copies from pageable memory, you guarantee that the copy speed will be bounded by the lower of the PCIE transfer speed and the system front-side bus speeds. A large disparity in bandwidth between these buses in some systems ensures that page-locked host memory enjoys roughly a twofold performance advantage over standard pageable memory when used for copying data between the GPU and the host. But even in a world where PCI Express and front-side bus speeds were identical, pageable buffers would still incur the overhead of an additional CPU-managed copy.

但是，您应该抵制简单地在 `malloc` 上进行搜索和替换以将每个调用转换为使用 `cudaHostAlloc()` 的诱惑。使用固定内存是一把双刃剑。通过这样做，您有效地选择了退出虚拟内存的所有优秀功能。具体来说，运行应用程序的计算机需要为每个页面锁定缓冲区提供可用的物理内存，因为这些缓冲区永远无法换出到磁盘。这意味着您的系统将比您坚持标准的 `malloc()` 调用更快地耗尽内存。这不仅意味着您的应用程序可能会在物理内存较少的机器上开始出现故障，而且还意味着您的应用程序可能会影响系统上运行的其他应用程序的性能。
However, you should resist the temptation to simply do a search-and-replace on `malloc` to convert every one of your calls to use `cudaHostAlloc()`. Using pinned memory is a double-edged sword. By doing so, you have effectively opted out of all the nice features of virtual memory. Specifically, the computer running the application needs to have available physical memory for every page-locked buffer, since these buffers can never be swapped out to disk. This means that your system will run out of memory much faster than it would if you stuck to standard `malloc()` calls. Not only does this mean that your application might start to fail on machines with smaller amounts of physical memory, but it means that your application can affect the performance of other applications running on the system.

### 10.3 CUDA Streams

CUDA 流表示按特定顺序执行的 GPU 操作队列。我们可以将内核启动、内存复制和事件启动和停止等操作添加到流中。将操作添加到流中的顺序指定了它们将执行的顺序。您可以将每个流视为 GPU 上的一个任务，这些任务有机会并行执行。
A CUDA stream represents a queue of GPU operations that get executed in a specific order. We can add operations such as kernel launches, memory copies, and event starts and stops into a stream. The order in which operations are added to the stream specifies the order in which they will be executed. You can think of each stream as a task on the GPU, and there are opportunities for these tasks to execute in parallel.

### 10.4 Using a Single CUDA Stream

