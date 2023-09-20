# Inspired by: https://github.com/pytorch/pytorch/issues/50334

import torch
from torch import Tensor

def torch_interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indexes = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indexes = torch.clamp(indexes, 0, len(m) - 1)

    return m[indexes] * x + b[indexes]

def torch_interp_Nd(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:,1:] - fp[:,:-1]) / (xp[:,1:] - xp[:,:-1])  #slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]) )

    indexes = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  #torch.ge:  x[i] >= xp[i] ? true: false
    indexes = torch.clamp(indexes, 0, m.shape[-1] - 1)

    line_idx = torch.arange(len(indexes), device=indexes.device).view(-1, 1)
    line_idx = line_idx.expand(indexes.shape)
    return m[line_idx, indexes].mul(x) + b[line_idx, indexes]

def torch_conv(in1, in2):
  in1 = in1.unsqueeze(0).unsqueeze(0)
  in2 = in2.unsqueeze(0).unsqueeze(0)
  in1_flip = torch.flip(in1, (0, 2)) 
  out = torch.conv1d(in1_flip, in2, padding=in1.shape[2]) 
  out = out[0, 0, in1.shape[2]+1:]
  out = torch.flipud(out)
  return out
  
def torch_conv_batch(in1, in2):
  b, s, t = in1.shape
  o = torch.conv1d(torch.flip(in1, (0, 2)), in2, padding=in1.shape[2], groups=s)
  o = o[:, :, in1.shape[2]+1:]
  o = torch.flip(o, (0, 2))
  return o