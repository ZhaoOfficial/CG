# Physically Based Rendering for Ray-Tracing (PBRT)

This home-made PBRT combines `pbrt-v3` and `pbrt-v4`. It is based on official version, but also has my own thoughts about implementation.

The core codes are in `src/`, `include/`, and `main`. The notes and documentations are in `doc/`. In the notes, we will briefly introduce some modern `C++` techniques and detailedly describe ray-tracing algorithm.

## Check List

### Chapter 1 Introduction

- [x] 1.1 Literate Programming
- [ ] 1.2 Photorealistic Rendering and The Ray-Tracing Algorithm
- [ ] 1.3 pbrt: System Overview
- [ ] 1.4 Parallelization of pbrt
- [ ] 1.5 How to Proceed through This Book
- [ ] 1.6 Using and Understanding the Code
- [ ] 1.7 A Brief History of Physically Based Rendering

### Chapter 2 Geometry and Transformations

- [x] 2.1 Coordinate Systems
- [ ] 2.2 Vectors
- [ ] 2.3 Points
- [ ] 2.4 Normals
- [ ] 2.5 Rays
- [ ] 2.6 Bounding Boxes
- [ ] 2.7 Transformations
- [ ] 2.8 Applying Transformations
- [ ] 2.9 Animating Transformations
- [ ] 2.10 Interactions

### Chapter 3 Shapes

- [ ] 3.1 Basic Shape Interface
- [ ] 3.2 Sphere
- [ ] 3.3 Cylinders
- [ ] 3.4 Disks
- [ ] 3.5 Other Quadratics
- [ ] 3.6 Triangle Meshes
- [ ] 3.7 Curves
- [ ] 3.8 Subdivision Surfaces
- [ ] 3.9 Managing Rounding Error