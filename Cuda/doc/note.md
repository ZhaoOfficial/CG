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

Page 26

## Chapter 4 Parallel Programming in CUDA C

### 4.2 CUDA Parallel Programming

Page 40







