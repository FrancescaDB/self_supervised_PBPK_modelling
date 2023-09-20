import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from scipy.stats import pearsonr
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import DynPETDataset
from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from utils.utils_kinetic import PET_2TC_KM_batch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction, apply_final_activation
from utils.utils_logging import log_slice, log_curves, mask_data
from utils.utils_torch import torch_interp_Nd
from utils.set_root_paths import root_path, root_checkpoints_path
import utils.similaritymeasures_torch as similaritymeasures_torch

torch.cuda.empty_cache()

if not torch.cuda.is_available():   
  current_gpu = None    
  machine = "cpu"
  print("*** ERROR: no GPU available ***")
else:
  machine = "cuda:0"
  current_gpu = [0]

class SpaceTempUNet(pl.LightningModule):
  
  def __init__(self, config):

    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(SpaceTempUNet, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.config["output_size"] = 4    # This is the number of output channels
    self.config["mask_loss"] = True
    self.config["multi_clamp_params"] = {"k1": (0.01, 2), "k2": (0.01, 3), "k3": (0.01, 1), "Vb": (0, 1)}
    # The paper was developed with config["patch_size"] = 112. However in this was some regions of the body are outsie the FOV (for example the arms of part of the belly or the back).
    # While config["patch_size"] = 112 doesn't reduce the performance of the training, using a larger patch_size will allow to infere complete 3D volumes without missing parts. 
    self.config["patch_size"] = 112 
    self.config["use_pearson_metric"] = False     # Setting it to True may slow down the training 
    print(self.config)

    if self.config["use_spatio_temporal_unet"]:
      self.model = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    frame_duration = [10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60, 120, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    self.frame_duration = np.array(frame_duration) / 60  # from s to min
    self.frame_duration_batch = torch.from_numpy(np.array(self.frame_duration)).unsqueeze(-1).repeat(1, self.config["patch_size"]*self.config["patch_size"]).to(machine)

  def setup(self, stage): 
    self.stage = stage
    self.patch_size = self.config["patch_size"]
    
    if stage == "fit":
      self.train_dataset = DynPETDataset(self.config, "train")
      self.idif_train_set = self.train_dataset.idif

      self.val_dataset = DynPETDataset(self.config, "validation")
      self.idif_val_set = self.val_dataset.idif

      self.t = self.train_dataset.t.to(machine)
      self.time_stamp = self.train_dataset.time_stamp.to(machine)
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    if stage == "test":
      self.test_dataset = DynPETDataset(self.config, "test")
      self.idif_test_set = self.test_dataset.idif
      self.t = self.test_dataset.t.to(machine)
      self.time_stamp = self.test_dataset.time_stamp.to(machine)
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    
  def forward(self, x):
    x = self.model(x)
    return x
  
  def loss_function(self, pred_TAC, real_TAC):        
    loss = similaritymeasures_torch.mse(pred_TAC.to(machine).double(), real_TAC.to(machine).double())
    return loss
  
  def train_dataloader(self):
      train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)
      return train_loader

  def val_dataloader(self):
      val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=0)
      return val_loader
  
  def test_dataloader(self):
      # If batch_size!=1 test_set and test_epoch_end may not work as expected
      test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
      return test_loader
    
  def accumulate_func(self, batch, logits, return_value=None):
      p = batch[0]              # len(p) = b
      TAC_batch = batch[2]      # [b, 62, w, h]

      b, t, h, w = TAC_batch.shape
      TAC_batch = torch.reshape(TAC_batch, [b, t, h*w])               # measurements
      TAC_pred_batch = torch.zeros_like(TAC_batch)

      kinetic_params = apply_final_activation(logits, self.config)          #  [b, 4, 1, 112, 112]
      kinetic_params = torch.reshape(kinetic_params, [b, self.config["output_size"], h*w, 1])   # [b, 4, w*h, 1]
      kinetic_params = kinetic_params.repeat(1, 1, 1, len(self.t))  #  [b, 4, w*h, 600]

      logits = torch.reshape(logits, [b, self.config["output_size"], h*w, 1])   # predicted params [b, 4, 1, 112, 112] --> [b, 4, w*h, 1]
      
      loss = 0

      metric_mse = 0
      metric_mae = 0
      metric_abc = 0
      cosine_sim = 0
      r2 = 0
      chi2 = 0
      pearson = 0
      pearson_p = 0
      counts_pearson = 0

      if not h*w == self.patch_size*self.patch_size:
        print("ERROR: the wrong patch size is used!")
        return

      time_stamp_batch = self.time_stamp_batch[:, 0, :].permute((1, 0))
      for i in range(b):
        current_TAC_batch = TAC_batch[i, :, :]
        current_kinetic_params = kinetic_params[i, :, :, :]
        current_TAC_pred_batch_long, _ = self.make_curve(current_kinetic_params, p[i])
        
        # Fix the mismatch in time: input has 62 time points, model has 600
        current_TAC_pred_batch_long = current_TAC_pred_batch_long[0, :, :].permute((1, 0))
        current_TAC_pred_batch = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], current_TAC_pred_batch_long)
        current_TAC_pred_batch = current_TAC_pred_batch.permute((1, 0))

        if self.config["mask_loss"]:    # More details about this can be found in utils/utils_logging.py
          AUC = torch.trapezoid(current_TAC_batch, time_stamp_batch, dim=0)
          maskk = AUC > 10
          maskk = maskk * 1
          mask = maskk.repeat(62, 1)
          current_TAC_batch = torch.multiply(current_TAC_batch, mask)
          current_TAC_pred_batch = torch.multiply(current_TAC_pred_batch, mask)
        
        TAC_pred_batch[i, :, :] = current_TAC_pred_batch
        if return_value is None or return_value == "Loss":  
          loss += self.loss_function(current_TAC_pred_batch, current_TAC_batch)

        if return_value is None or return_value == "Metric":  
          if self.config["mask_loss"]:
              square = torch.square(current_TAC_pred_batch.to(machine) - current_TAC_batch.to(machine))
              absolute = torch.abs(current_TAC_pred_batch.to(machine) - current_TAC_batch.to(machine))
              metric_mse += torch.sum(square).item() / len(mask[mask>0])
              metric_mae += torch.sum(absolute).item() / len(mask[mask>0])

              cosine_sim_slice = torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(machine), current_TAC_batch.to(machine), 0)
              cosine_sim += torch.sum(cosine_sim_slice).item() / len(maskk[maskk>0])

              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(machine))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.sum(chi2_slice).item() / len(maskk[maskk>0])
          else:
              metric_mse += similaritymeasures_torch.mse(current_TAC_pred_batch.to(machine), current_TAC_batch.to(machine)).item()
              metric_mae += similaritymeasures_torch.mae(current_TAC_pred_batch.to(machine), current_TAC_batch.to(machine)).item()
              cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(current_TAC_pred_batch.to(machine), current_TAC_batch.to(machine), 0)).item()

              square = torch.square(current_TAC_pred_batch.to(machine) - current_TAC_batch.to(machine))
              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_TAC_batch.to(machine))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.mean(chi2_slice).item()
          
          # The following metrics are the same independently from self.config["mask_loss"]
          r2 += similaritymeasures_torch.r2(current_TAC_pred_batch.to(machine), current_TAC_batch.to(machine)).item()

          if self.config["use_pearson_metric"]:
              for j in range(h*w):
                  current_pred_TAC = current_TAC_pred_batch[:, j]
                  current_real_TAC = current_TAC_batch[:, j]
                  if self.config["mask_loss"]: 
                      if torch.sum(current_pred_TAC) == 0 or torch.sum(current_real_TAC) == 0:
                          continue
                      else:
                          current_pearson, current_p = pearsonr(current_pred_TAC.cpu().detach().numpy(), current_real_TAC.cpu().detach().numpy())
                          pearson += current_pearson
                          pearson_p += current_p
                          counts_pearson += 1

      loss = loss/b           # Use b instead od self.config["batch_size"] to accomodate for batches of different size (like the last one)
      loss_dict =  {"loss": loss}

      metric_mse = metric_mse/b
      metric_mae = metric_mae/b
      metric_abc = metric_abc/b
      cosine_sim = cosine_sim/b
      r2 = r2/b
      if self.config["use_pearson_metric"]:
        pearson_p = pearson_p/counts_pearson
        pearson = pearson/counts_pearson
      
      metric_dict = {"mse": metric_mse, "mae": metric_mae, "cosine_sim": cosine_sim, "r2": r2}
      if self.config["use_pearson_metric"]:
        metric_dict["pearson_corr"] = pearson
        metric_dict["pearson_p"] = pearson_p
      
      TAC_pred_batch = torch.reshape(TAC_pred_batch, [b, 1, t, h, w])

      if return_value is None:
        return loss_dict, metric_dict, TAC_pred_batch
      elif return_value == "Loss":
        return loss_dict
      elif return_value == "Metric":
        return metric_dict, TAC_pred_batch

  def accumulate_loss(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Loss")
  
  def accumulate_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Metric")
  
  def accumulate_loss_and_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value=None)

  def make_curve(self, kinetic_params, patient):

    # Prepare the IDIF --> can't be moved outside the for loop because the patient can change inside the batch
    if self.stage == "fit" and patient in self.idif_train_set.keys():
      current_idif = self.idif_train_set[patient]
    elif self.stage == "fit" and patient in self.idif_val_set.keys():
      current_idif = self.idif_val_set[patient]
    elif self.stage == "test" and patient in self.idif_test_set.keys():
      current_idif = self.idif_test_set[patient]
    else: 
      print("ERROR: IDIF of patient "+str(patient)+" not found !")
      return
    idif_batch = current_idif.repeat(1, self.patch_size*self.patch_size, 1)

    # Prepare the kinetic parameters
    kinetic_params = kinetic_params.permute((1, 0, 2))
    k1 = kinetic_params[:, 0, :].unsqueeze(1)
    k2 = kinetic_params[:, 1, :].unsqueeze(1)        
    k3 = kinetic_params[:, 2, :].unsqueeze(1)
    Vb = kinetic_params[:, 3, :].unsqueeze(1)

    # Compute the TAC
    current_pred_curve = PET_2TC_KM_batch(idif_batch.to(machine), self.t_batch.to(machine), k1, k2, k3, Vb)
    
    return current_pred_curve, None

  def training_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1))   # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 4, 1, w, h]
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)
    self.log('train_loss', loss_dict["loss"].item(), on_step=False, on_epoch=True)

    if batch_idx % 20 == 0:

      if self.config["mask_loss"]:
        TAC_mes_batch, TAC_pred_batch, logits_params = mask_data(TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)
                 
      fig = log_curves(TAC_mes_batch[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      wandb.log({"TAC (training batch: "+str(batch_idx)+")": wandb.Image(fig)})
      plt.close()

      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, TAC_mes_batch, kinetic_params)
      wandb.log({"Slice (training batch: "+str(batch_idx)+")": wandb.Image(fig)})
      plt.close()

    return {"loss": loss_dict["loss"]}

  def validation_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 4, 1, w, h]
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)

    self.log('val_loss', loss_dict["loss"].item(), on_step=False, on_epoch=True)
    self.log_dict(metric_dict, on_step=False, on_epoch=True)

    # Prepare data to log
    if batch_idx % 5 == 0:

      if self.config["mask_loss"]:
        TAC_mes_batch, TAC_pred_batch, logits_params = mask_data(TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)

      # Log TAC                
      fig = log_curves(TAC_mes_batch[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      fig_curve = {"TAC (validation batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.close()

      # Log slices
      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, TAC_mes_batch, kinetic_params)
      fig_slice = {"Slice (validation batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.close()

      return {"fig_slice": fig_slice, "fig_curve": fig_curve}
    else:
      return {'val_loss': loss_dict["loss"].item(), "fig_slice": None, "fig_curve": None}
  
  def validation_epoch_end(self, outputs):
    for o in outputs:
      if not o["fig_slice"] is None:  wandb.log(o["fig_slice"])
      if not o["fig_curve"] is None:  wandb.log(o["fig_curve"])
    return
  
  def test_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    patients_in_batch = batch[0]
    slices_in_batch = batch[1]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    x = torch.nn.functional.pad(TAC_mes_batch, (0,0,0,0,1,1), "replicate")   # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 4, 1, w, h]
    metric_dict, TAC_pred_batch = self.accumulate_metric(batch=batch, logits=logits_params)
    kinetic_params = apply_final_activation(logits_params, self.config)

    if self.config["mask_loss"]:
      TAC_mes_batch, TAC_pred_batch, kinetic_params = mask_data(TAC_mes_batch, TAC_pred_batch, kinetic_params, self.time_stamp, patch_size=self.patch_size)

    # Save predictions
    current_run_name = wandb.run.name
    resume_run_name = os.path.split(os.path.split(self.trainer.ckpt_path)[0])[1]
    self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, resume_run_name, root_checkpoints_path, self.trainer.ckpt_path)
    to_save = [patients_in_batch, slices_in_batch, kinetic_params]
    s = int(slices_in_batch.item())
    torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".pt"))

    # Prepare data to log
    if batch_idx % 50 == 0:
      if not len(slices_in_batch) == 1: s = slices_in_batch

      # Log TAC             
      log_curves(TAC_mes_batch[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      plt.savefig(os.path.join(self.img_path, "TAC_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

      # Log slices
      log_slice(self.config, TAC_mes_batch, kinetic_params)
      plt.savefig(os.path.join(self.img_path, "slices_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

    return {"patients_in_batch": patients_in_batch, "slices_in_batch": slices_in_batch, "metric_dict": metric_dict}
  
  def test_epoch_end(self, outputs):
    run_name = os.path.split(os.path.split(self.pd_path)[0])[1]
    summary = dict()

    for o in outputs:
      metric_dict = o["metric_dict"]
      patients_in_batch = o["patients_in_batch"]
      slices_in_batch = o["slices_in_batch"]
      for i in range(len(patients_in_batch)):
        p = patients_in_batch[i]
        if not p in summary.keys(): summary[p] = dict()
        for j in range(len(slices_in_batch)):
          s = int(slices_in_batch[j].item())
          if patients_in_batch[j] == p:
            summary[p][s] = dict()
            summary[p][s]["MSE"] = metric_dict["mse"]
            summary[p][s]["MAE"] = metric_dict["mae"]
            summary[p][s]["CosineSim"] = metric_dict["cosine_sim"]

    for p in summary.keys():
      current_df = pd.DataFrame.from_dict(summary[p])
      # This file contains the metrics per slice. It allows to identify slices with bad peformance. 
      # It is also used during evaluation phase to compute the metrics on the whole dataset
      current_df.to_excel(os.path.join(self.pd_path, p+"_metric_per_slice_"+run_name+".xlsx"))

    # Reconstruct the 3D kinetic parameters volumes
    reconstruct_prediction(self.pt_path, self.nifty_path)
    return 

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100], gamma=0.5, verbose=True)
    return [optimizer], [scheduler]
  
def train_unet(resume_path=None, enable_testing=False):
  # Set up Weights&Biases Logger
  wandb_logger = WandbLogger(project="DynPET_MICCAI_release", config=os.path.join(root_path, "config.yaml"))
  
  unet = SpaceTempUNet(wandb.config)

  # Set up the checkpoints
  checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)
  print("Checkpoints will be saved in: ", checkpoint_path)
  checkpoint_callback = ModelCheckpoint(  monitor="val_loss",
                                          dirpath=checkpoint_path,
                                          save_top_k=1,
                                          mode="min",
                                          every_n_epochs=5,
                                          save_last=True,
                                      )
  early_stop_callback = EarlyStopping(  monitor="val_loss", 
                                        min_delta=0, 
                                        patience=25,
                                        verbose=True, 
                                        mode="min",
                                        check_finite=True)

  trainer = pl.Trainer(gpus=current_gpu,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=wandb_logger,
                        resume_from_checkpoint=resume_path
                    )
  
  trainer.fit(unet)

  if enable_testing:
    trainer.test(ckpt_path="best")
    trainer.test(ckpt_path="last")
  
  # Close Weights&Biases Logger 
  wandb.finish()

def test_unet(checkpoint_path=None):
  # Set up Weights&Biases Logger
  wandb_logger = WandbLogger(project="DynPET_MICCAI_release", config=os.path.join(root_path, "config.yaml"))

  if checkpoint_path is None:
    checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.config["saved_checkpoint"])
  print("Testing on", checkpoint_path)
  
  unet = SpaceTempUNet(wandb.config)

  trainer = pl.Trainer(gpus=current_gpu,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        logger=wandb_logger,
                      )
  
  trainer.test(unet, ckpt_path=checkpoint_path)
  
  # Close Weights&Biases Logger 
  wandb.finish()

if __name__ == '__main__':
  
  ### TRAIN ###
  train_unet(resume_path=None, enable_testing=False)

  ### TEST ###
  # test_unet()
  