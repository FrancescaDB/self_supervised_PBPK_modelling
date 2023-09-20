import torch
import matplotlib.pyplot as plt
from utils.custom_cmap import cmap

labels_to_organ = {0: "background", 3: "liver", 4: "lung_1", 5: "lung_2", 6: "kidney_1", 7: "kidney_2", 10: "heart", 11: "aorta", 13: "spleen", 16: "unknown", 19: "bones"}

def log_slice(config, PET_measurements, K_prediction):
  b = PET_measurements.shape[0]
  fig, ax = plt.subplots(nrows=b, ncols=7, figsize=(4*7,4*b))
  vmax_list = [1, 1, 0.6, 0.4, 4]
  if b == 1:
    im_0 = ax[0].imshow(PET_measurements[0, 0, 10, :, :].to("cpu"), vmax=torch.max(PET_measurements[0, 0, 10, :, :].to("cpu")).item()/2)
    im_1 = ax[1].imshow(PET_measurements[0, 0, -1, :, :].to("cpu"), vmax=torch.max(PET_measurements[0, 0, -1, :, :].to("cpu")).item()/2)
    plt.colorbar(im_0, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im_1, ax=ax[1], fraction=0.046, pad=0.04)
    for i in range(config["output_size"]):
      K_slice = K_prediction[0, i, 0, :, :]
      im = ax[i+2].imshow(K_slice.to("cpu").detach().numpy(), cmap=cmap, vmin=0, vmax=vmax_list[i])
      plt.colorbar(im, ax=ax[i+2], fraction=0.046, pad=0.04)
    K_slice = K_prediction[0, 0, 0, :, :]/K_prediction[0, 1, 0, :, :]
    im_2 = ax[6].imshow(K_slice.to("cpu").detach().numpy(), cmap=cmap, vmin=0, vmax=vmax_list[4])
    ax[0].set_title("PET (t=10)")
    ax[1].set_title("PET (t=60)")
    ax[2].set_title("K1")
    ax[3].set_title("k2")
    ax[4].set_title("k3")
    ax[5].set_title("Vb")
    plt.colorbar(im_2, ax=ax[6], fraction=0.046, pad=0.04)
  else:
    for j in range(b):
      im_0 = ax[j, 0].imshow(PET_measurements[j, 0, 10, :, :].to("cpu"), vmax=torch.max(PET_measurements[j, 0, 60, :, :].to("cpu")).item()/2)
      im_1 = ax[j, 1].imshow(PET_measurements[j, 0, -1, :, :].to("cpu"), vmax=torch.max(PET_measurements[j, 0, -1, :, :].to("cpu")).item()/2)
      plt.colorbar(im_0, ax=ax[j, 0], fraction=0.046, pad=0.04)
      plt.colorbar(im_1, ax=ax[j, 1], fraction=0.046, pad=0.04)
      ax[j, 0].set_ylabel("Element "+str(j)+" in batch")
      for i in range(config["output_size"]):
          K_slice = K_prediction[j, i, 0, :, :]
          im = ax[j, i+2].imshow(K_slice.to("cpu").detach().numpy(), cmap=cmap, vmin=0, vmax=vmax_list[i])
          plt.colorbar(im, ax=ax[j, i+2], fraction=0.046, pad=0.04)
      K_slice = K_prediction[j, 0, 0, :, :]/K_prediction[j, 1, 0, :, :]
      im_2 = ax[j, 6].imshow(K_slice.to("cpu").detach().numpy(), cmap=cmap, vmin=0, vmax=vmax_list[4])
      plt.colorbar(im_2, ax=ax[j, 6], fraction=0.046, pad=0.04)
    ax[0, 0].set_title("PET (t=10)")
    ax[0, 1].set_title("PET (t=60)")
    ax[0, 2].set_title("K1")
    ax[0, 3].set_title("k2")
    ax[0, 4].set_title("k3")
    ax[0, 5].set_title("Vb")
    ax[0, 6].set_title("K1/k2")
  plt.tight_layout()
  return fig

def log_curves(PET_measurements, PET_prediction, t, time_stamp, current_epoch):
  b = PET_measurements.shape[0]
  fig, ax = plt.subplots(nrows=b, ncols=6, figsize=(2*5, 2*b))
  if b == 1:
    for i in range(5):  
      ax[i].plot(time_stamp, PET_measurements[0, 0, :, (i+2)*10, (i+2)*10], color="tab:red", label="mes")
      if current_epoch > 0: ax[i].plot(t, PET_prediction[0, 0, :, (i+2)*10, (i+2)*10], color="tab:green", label="pred")
      ax[i].set_title("Voxel: ["+str((i+2)*10)+", "+str((i+2)*10)+"]")
      # ax[i].set_ylim([0, 10])
    ax[4].legend()
    im = ax[5].imshow(torch.nn.functional.cosine_similarity(torch.from_numpy(PET_measurements[0, 0, :, :, :]), torch.from_numpy(PET_prediction[0, 0, :, :, :]), 0), vmin=-1, vmax=1)
    ax[5].set_title("Correlation")
    plt.colorbar(im, ax=ax[5], fraction=0.046, pad=0.04)
  else:
    for j in range(PET_measurements.shape[0]):
      for i in range(5):  
        ax[j, i].plot(time_stamp, PET_measurements[j, 0, :, (i+2)*10, (i+2)*10],  color="tab:red", label="mes")
        if current_epoch > 0: ax[j, i].plot(t, PET_prediction[j, 0, :, (i+2)*10, (i+2)*10], color="tab:green", label="pred")
        # ax[j, i].set_ylim([0, 10])
        ax[0, i].set_title("Voxel: ["+str((i+2)*10)+", "+str((i+2)*10)+"]")
      ax[j, 0].set_ylabel("Element "+str(j)+" in batch")
      im = ax[j, 5].imshow(torch.nn.functional.cosine_similarity(torch.from_numpy(PET_measurements[j, 0, :, :, :]), torch.from_numpy(PET_prediction[j, 0, :, :, :]), 0), vmin=-1, vmax=1)
      ax[0, 5].set_title("Correlation")
      plt.colorbar(im, ax=ax[j, 5], fraction=0.046, pad=0.04)
    ax[0, 4].legend()
  plt.tight_layout()
  return fig

def mask_data(TAC_mes_batch, TAC_pred_batch, y_pred, time_stamp, patch_size):
  """
  This function mask (i.e. set to zero) all the pixels in which the Area Under the Curve of 
  the TAC is lower than a threshold (empirically set to 10). The assumption is that TACs with AUC < 10
  belong to voxels in the air, and therefore we don't need to optimize for them. 
  The same concept is applied in main.accumulate_func
  """
  b = TAC_mes_batch.shape[0]

  # Prepare TAC
  time_stamp_batch = time_stamp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  time_stamp_batch = time_stamp_batch.permute(1, 2, 0, 3, 4)
  time_stamp_batch = time_stamp_batch.repeat(b, 1, 1, patch_size, patch_size)

  AUC = torch.trapezoid(TAC_mes_batch, time_stamp_batch, dim=2)
  maskk = AUC > 10
  maskk = maskk * 1
  maskk = maskk.unsqueeze(1)
  mask = maskk.repeat(1, 1, 62, 1, 1)
  TAC_mes_batch = torch.multiply(TAC_mes_batch, mask)
  TAC_pred_batch = torch.multiply(TAC_pred_batch, mask)

  # Prepare predictions
  mask = maskk.repeat(1, 4, 1, 1, 1)
  y_pred = torch.multiply(y_pred, mask)

  return TAC_mes_batch, TAC_pred_batch, y_pred