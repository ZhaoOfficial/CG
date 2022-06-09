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

块中的线程数不能超过设备属性的 `maxThreadsPerBlock` 字段指定的值。
The number of threads in a block cannot exceed the value specified by the `maxThreadsPerBlock` field of the device properties.

```c
// for GeForce RTX 2060
Maximum number of threads per block: 1024
Maximum sizes of each dimension of a block: 1024 x 1024 x 64
Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
```



