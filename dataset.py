import os
import torch
import numpy as np
import glob
import SimpleITK as sitk
from monai.data import CacheDataset
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_kinetic import PET_2TC_KM
from utils.utils_torch import torch_interp_1d
from utils.set_root_paths import root_data_path

class DynPETDataset(CacheDataset):

    def __init__(self, config, dataset_type): 
        # Enforce determinism
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Read global config
        self.config = config
        self.dataset_type = dataset_type
        self.patch_size = self.config["patch_size"]

        # Create config for each dataset type from global config
        self.train_config = {"patient_list": self.config["patient_list"]["train"], "slices_per_patient": int(self.config["slices_per_patient_train"])}
        self.val_config = {"patient_list": self.config["patient_list"]["validation"], "slices_per_patient": int(self.config["slices_per_patient_val"])}
        if self.config["slices_per_patient_test"] is "None":
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": 500}     # Take all the slices from the test patients
        else: 
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": self.config["slices_per_patient_test"]}

        # Select the correct config
        self.idif = dict()
        self.data = list()
        if dataset_type == "train":
            self.build_dataset(self.train_config)
        elif dataset_type == "validation":
            self.build_dataset(self.val_config)
        elif dataset_type == "test":
            self.build_dataset(self.test_config)
        else: 
            print("ERROR: dataset type not supported!")
            return

    def __getitem__(self,idx):                  
        return self.data[idx]

    def __len__(self):
        return int(self.length)
    
    def build_dataset(self, current_config):
        self.current_dataset_size = current_config["slices_per_patient"] * len(current_config["patient_list"])     
        print("Creating dataset", self.dataset_type, ":", current_config)
            
        self.patient_list = current_config["patient_list"]
        for p in self.patient_list:
            self.load_txt_data(p)

        # Load exiting data, if possible
        load_data = self.load_data()
        if load_data is None:   
            self.read_dynpet()
            print("Dataset", self.dataset_type, "was saved in", self.save_data_folder)
        else:                   
            self.data = load_data
        
        self.length = len(self.data) 
        print("Dataset", self.dataset_type, "has", self.current_dataset_size, "slices!\n")
        return

    def load_txt_data(self, patient):
        tac_txt_path = os.path.join(root_data_path, "DynamicPET/TAC", "DynamicFDG_"+patient+"_TAC.txt")
        idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", "DynamicFDG_"+patient+"_IDIF.txt")

        # Read acquisition time
        data = pd.read_csv(tac_txt_path, sep="\t")
        data['start[seconds]'] = data['start[seconds]'].apply(lambda x: x/60)
        data['end[kBq/cc]'] = data['end[kBq/cc]'].apply(lambda x: x/60)
        data.rename(columns={"start[seconds]": "start[minutes]"}, inplace=True)
        data.rename(columns={"end[kBq/cc]": "end[minutes]"}, inplace=True)
        time_stamp = data['start[minutes]'].values
        self.time_stamp = torch.Tensor(np.around(time_stamp, 2))

        # Define interpolated time axis which is required to run the convolution
        step = 0.1
        self.t = torch.Tensor(np.arange(self.time_stamp[0], self.time_stamp[-1], step))

        # Read IDIF and interpolate it ì
        rolling_window = 1
        data = pd.read_csv(idif_txt_path, sep="\t").rolling(rolling_window).mean()
        idif_sample_time = torch.Tensor(data["sample-time[minutes]"])
        idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.idif[patient] = torch_interp_1d(self.t, idif_sample_time, idif)
        return 

    def read_dynpet(self): 
        data_list = list()
        for patient in self.patient_list:
            patient_folder = glob.glob(os.path.join(root_data_path, "DynamicPET", "*DynamicFDG_"+patient))[0]

            # When using self.config["slices_per_patient_*"] --> probably the selection of self.slice can be shortened a little bit
            self.slices_per_patients = int(self.current_dataset_size / len(self.patient_list))

            if self.current_dataset_size == 1: 
                # When using only one slice per patient (for example during debugging), the location of the slice is hard-coded (it should select a slice with kidneys)
                self.slices = [212]
                print("\tPatient: " + str(patient) + "; N_slices=" + str(len(self.slices)) + "/1 ; slices:", self.slices)
            else: 
                # When config["slices_per_patient_*"]>1 (and therefore self.current_dataset_size > 1), the slices are selected within a ROI defined by a bouding box (bb). We used a bb
                # including the lungs and the bladder. In this way we didn't considered the head (because of the movement over the acquisition) and the legs (which are not very informative).
                # The use of the bb is not mandatory.  
                bb_path = patient_folder+"/NIFTY/Resampled/bb.nii.gz"
                bb_ = sitk.GetArrayFromImage(sitk.ReadImage(bb_path))

                # First, the indexes of the slices are picked homogeneously withing the indexes of the bb
                indexes = np.nonzero(bb_)
                top = indexes[0][-1]
                bottom = indexes[0][0]
                step = np.floor((top - bottom) / self.slices_per_patients)
                if step == 0:       # This happens if the bb is smaller than the self.slices_per_patients (top - bottom < self.slices_per_patients)
                    step = 1
                hom_pick = torch.arange(bottom, top, step)

                # If the homogeneous pick is much bigger than the expected dataset size, the pick is reduced by lowering the sampling frequency at borders of bb.
                # The underlying assumption is that the most informative region is the center of the bb.
                # This step can be omitted (and just use pick = hom_pick)
                if len(hom_pick)-self.slices_per_patients > 50:
                    center_slice = int(len(hom_pick)/2)
                    a = int((len(hom_pick)-self.slices_per_patients) * 2 / 3)
                    new_step = int(step+1)
                    hom_pick_short = torch.concat((hom_pick[:center_slice-a][::new_step], hom_pick[center_slice-a:center_slice+a], hom_pick[center_slice+a:-1][::new_step]))
                    if len(hom_pick_short) > self.slices_per_patients: pick = hom_pick_short
                    else: pick = hom_pick
                else: 
                    pick = hom_pick

                # Final selection of the pick 
                if top - bottom < self.slices_per_patients:            
                    # All the slices in the bb can be selected
                    self.slices = hom_pick[0:self.slices_per_patients]
                elif self.dataset_type == "test":
                    # When testing, we always use the homogeneous pick
                    self.slices = hom_pick[0:self.slices_per_patients]
                else:
                    # In all the other cases, self.slices_per_patients are selected in the center of the bb
                    c = int(len(pick)/2)
                    s = int(self.slices_per_patients/2)
                    self.slices = pick[c-s:c+s+1]                       # Select N=self.slices_per_patients slices in mid of pick
                
                # len(self.slices) may not be exactly equal to self.slices_per_patients beacuse of numerical errors (their differece it's usually 1)
                print("\tPatient: " + str(patient) + "; N_slices=" + str(len(self.slices)) + "/" + str(top - bottom) + "; slices:", self.slices)

            # Load dynamic PET
            size = self.patch_size
            pet_list = glob.glob(patient_folder+"/NIFTY/Resampled/PET_*.nii.gz")
            data = list()
            current_data = torch.zeros((len(self.slices), len(pet_list), size, size))          
            for i in range(len(pet_list)):
                p = pet_list[i]
                current_pet = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(p))) 

                # Define borders of center crop
                slice_size = current_pet[0, :, :].shape
                slice_center = torch.tensor(slice_size)[0] / 2
                for j in range(len(self.slices)): 
                    slice = int(self.slices[j])
                    current_slice = current_pet[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]                
                    current_data[j, i, :, :] = current_slice/1000           # from Bq/ml to kBq/ml
            
            # Load label map
            # label_path = patient_folder+"/NIFTY/Resampled/labels.nii.gz"
            # current_label_data = torch.zeros((len(self.slices), size, size))          
            # label_ = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
            # slice_size = label_[0, :, :].shape
            # slice_center = torch.tensor(slice_size)[0] / 2
            # for j in range(len(self.slices)): 
            #     slice = int(self.slices[j])
            #     current_slice = label_[slice, int(slice_center)-int(size/2):int(slice_center)+int(size/2), int(slice_center)-int(size/2):int(slice_center)+int(size/2)]                
            #     current_label_data[j, :, :] = current_slice

            # for j in range(len(self.slices)):
            #     data.append([patient, self.slices[j], current_data[j, :, :], current_label_data[j, :, :]])
            
            for j in range(len(self.slices)):
                data.append([patient, self.slices[j], current_data[j, :, :]])
            
            data_list = data_list + data
        data_list = data_list[0:self.current_dataset_size]
        self.data = data_list
        torch.save(data_list, self.save_data_folder+"/data"+str(self.patient_list)+".pt")
        
        return data_list
    
    def load_data(self):
        # Define the location where the dataset is saved
        folder_name = self.dataset_type+"_N"+str(self.current_dataset_size)+"_P"+str(self.patch_size)
        if root_data_path == "/mnt/polyaxon/data1":
            self.save_data_folder = os.path.join("/home/lamp/Documents/Francesca/dynamicpet/local_dataset", folder_name)
        elif root_data_path == "/Volumes/polyaxon/data1":
            self.save_data_folder = os.path.join(root_data_path, "DynamicPET", "dataset_release", folder_name)
        elif root_data_path == "/data":
            self.save_data_folder = os.path.join(root_data_path, "DynamicPET", "dataset", folder_name)
        elif root_data_path == "/home/polyaxon-data/data1":
            self.save_data_folder = os.path.join("/home/guests/francesca_de_benetti/francesca/data/MICCAI_release/dataset", folder_name)

        # Create the folder if it doesn't exist
        if not os.path.exists(self.save_data_folder):       
            os.makedirs(self.save_data_folder)

        # If the dataset exists, load it and return it
        file_name = "data"+str(self.patient_list)+".pt"
        if os.path.exists(self.save_data_folder+"/"+file_name):
            data = torch.load(self.save_data_folder+"/"+file_name)
            return data
        else:
            print("\tWARNING: " + file_name + " doesn't exist!")
            return None
        
        # GOOD TO KNOW: self.save_data_folder and file_name are designed so that different datasets with different patients list, patch size or number of slices are saved separately and not overwritten.
        # This allows to save time when generating bigger datasets!