import torch
from utils.utils_torch import torch_conv, torch_conv_batch

# Use of step: https://dsp.stackexchange.com/questions/26079/why-does-the-amplitude-of-a-discrete-convolution-depend-on-the-time-step
def PET_2TC_KM(input_TAC, t, k1, k2, k3, Vb, step=0.1):
  # This is the 2-Tissue-Compartment Kinetic Model which describes what is measured by a PET scanner in one ROI (voxel, pixel, organ, ...) over time. 
  # It takes as input the Image Derived Input Function (IDIF) which represents the amount of radioactivity in the blood/plasma over time. 
  # The IDIF is calculated by averaging the TAC in the descending aorta and it is patient-specific. 
  # In this work it is considered to be the same for every region of the body. In other words, we don't take into account the fact that the peak of 
  # the IDIF occurs at different time point depending on the region of the body. This is conventionally included as a "delay" of the IDIF.
  a = input_TAC     # idif
  e = (k2+k3)*t
  b = k1 / (k2+k3) * (k3 + k2*torch.exp(-e))         # 2TC irreversible
  c = torch_conv(a, b) * step
  PET = (1-Vb) * c + Vb * a
  PET.requires_grad_()
  return PET

def PET_2TC_KM_batch(input_TAC, t, k1, k2, k3, Vb, step=0.1):
  a = input_TAC    # idif
  e = torch.multiply(k2+k3, t)
  b = k1 / (k2+k3) * (k3 + k2*torch.exp(-e))         # 2TC irreversible
  c = torch_conv_batch(a, b) * step

  # The permutations are required to get the expected shape
  Vb = Vb.permute((1, 0, 2))
  PET = (1-Vb) * c + Vb * a

  PET = PET.permute((0, 2, 1))
  PET.requires_grad_()
  return PET


  