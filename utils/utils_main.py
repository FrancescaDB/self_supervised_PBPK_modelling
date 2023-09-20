import numpy as np
import torch
import os
import glob
import SimpleITK as sitk
from natsort import natsorted

if not torch.cuda.is_available():   
  machine = "cpu"
else:
  machine = "cuda:0"

def apply_final_activation(logits, config):
  # Prepare the predicted k-values
  if config["final_activation"] == "auto": 
    kinetic_params = logits.to(machine)
  elif config["final_activation"] == "abs":  
    kinetic_params = torch.abs(logits.to(machine))
  elif config["final_activation"] == "clamp":  
    kinetic_params = torch.clamp(logits.to(machine), 0.01, 3)
  elif config["final_activation"] == "multi_clamp":  
    kinetic_params = torch.zeros_like(logits)
    kinetic_params[:, 0, :, :, :] = torch.clamp(logits[:, 0, :, :, :].to(machine), 
                                                      config["multi_clamp_params"]["k1"][0], config["multi_clamp_params"]["k1"][1]) 
    kinetic_params[:, 1, :, :, :] = torch.clamp(logits[:, 1, :, :, :].to(machine), 
                                                      config["multi_clamp_params"]["k2"][0], config["multi_clamp_params"]["k2"][1]) 
    kinetic_params[:, 2, :, :, :] = torch.clamp(logits[:, 2, :, :, :].to(machine), 
                                                      config["multi_clamp_params"]["k3"][0], config["multi_clamp_params"]["k3"][1]) 
    kinetic_params[:, 3, :, :, :] = torch.clamp(logits[:, 3, :, :, :].to(machine), 
                                                      config["multi_clamp_params"]["Vb"][0], config["multi_clamp_params"]["Vb"][1]) 
  else: 
    print("*** ERROR: invalid config['modality']: using abs ***")
    return
  return kinetic_params

def make_save_folder_struct(current_run_name, resume_run_name, root_checkpoints_path, trainer_ckpt_path, if_return_only=False):
  if current_run_name == resume_run_name:
      run_name = current_run_name
  else:
    run_name = resume_run_name

  if "last" in trainer_ckpt_path: 
    save_path = os.path.join(root_checkpoints_path, "output", run_name+"_last")
  else:    
    save_path = os.path.join(root_checkpoints_path, "output", run_name+"_best")
  img_path = os.path.join(save_path, "images")
  pd_path = os.path.join(save_path, "tables")
  pt_path = os.path.join(save_path, "raw_prediction")
  nifty_path = os.path.join(save_path, "nifty")
  if not if_return_only:
    if not os.path.exists(img_path):   os.makedirs(img_path)
    if not os.path.exists(pd_path):   os.makedirs(pd_path)
    if not os.path.exists(pt_path):   os.makedirs(pt_path)
    if not os.path.exists(nifty_path):   os.makedirs(nifty_path)

  return img_path, pd_path, pt_path, nifty_path

def reconstruct_prediction(pt_path, nifty_path, patient=None):
  # The arg patient can be set when the method is used to reconstruct one specific prediction volume offline
  # The default set-up (patient=None) reconstructs all the volumes available (i.e. all the volumes of the patients included in config["patient_list"]["test"])
  if patient is None:
    predictions_list = natsorted(glob.glob(pt_path+"/*.pt"))
  else:
    predictions_list = natsorted(glob.glob(pt_path+"/*"+patient+"*.pt"))

  reconstruct_prediction = dict()
  reconstruct_prediction["patients"] = list()
  reconstruct_prediction["predictions"] = None

  for path in predictions_list:
      prediction = torch.load(path)
      patients_in_batch = prediction[0]
      prameteric_img = prediction[2]

      reconstruct_prediction["patients"] += patients_in_batch
      if reconstruct_prediction["predictions"] is None:
          reconstruct_prediction["predictions"] = prameteric_img[:, :, 0, :, :].to(machine)
      else: 
          reconstruct_prediction["predictions"] = torch.concat((reconstruct_prediction["predictions"].to(machine), prameteric_img[:, :, 0, :, :].to(machine)))

  # Reconstruct prediction
  prediction = reconstruct_prediction['predictions'].cpu().detach().numpy()
  run_name = os.path.split(os.path.split(nifty_path)[0])[1]
  if patient is None:
    for p in np.unique(reconstruct_prediction['patients']):
      indexes = np.where(np.array(reconstruct_prediction['patients'])==p)
      for i in range(prediction.shape[1]):
        current_k_ = prediction[indexes, i, :, :]
        current_k = sitk.GetImageFromArray(current_k_[0, :, :, :])
        current_k.SetSpacing([2.5, 2.5, 2.5])
        sitk.WriteImage(current_k, os.path.join(nifty_path, str(p)+"_"+str(i)+"_"+run_name+".nii.gz"))
  else:
    indexes = np.where(np.array(reconstruct_prediction['patients'])==patient)
    for i in range(prediction.shape[1]):
      current_k_ = prediction[indexes, i, :, :]
      current_k = sitk.GetImageFromArray(current_k_[0, :, :, :])
      current_k.SetSpacing([2.5, 2.5, 2.5])
      sitk.WriteImage(current_k, os.path.join(nifty_path, str(patient)+"_"+str(i)+"_"+run_name+".nii.gz"))