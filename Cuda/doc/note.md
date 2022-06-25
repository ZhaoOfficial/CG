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
