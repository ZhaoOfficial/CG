from typing import Callable

import torch

# tuple for immutable
C = (
    # 1
    (
        0.28209479177387814,
    ),
    # 2
    (
        -0.48860251190291987,
         0.48860251190291987,
        -0.48860251190291987,
    ),
    # 3
    (

    )
)

def gen_shs(num_degrees: int) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    """"""
    assert isinstance(num_degrees, int), "num_degrees must be a integer."
    assert 1 <= num_degrees <= 7, "only first 1 ~ 8 degree of spherical harmonics are supported."

    sh_funcs = []
    sh_funcs.append(lambda  : C[0][0])
    if num_degrees <= 1:
        return
    sh_funcs.append(lambda y: C[1][0] * y)
    sh_funcs.append(lambda z: C[1][1] * z)
    sh_funcs.append(lambda x: C[1][2] * x)
    if num_degrees <= 2:
        return
    if num_degrees <= 3:
        return
    if num_degrees <= 4:
        return
    if num_degrees <= 5:
        return
    if num_degrees <= 6:
        return
    if num_degrees <= 7:
        return

    return sh_funcs
