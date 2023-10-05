import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils.utils_logging import labels_to_organ
from utils.utils_kinetic import PET_2TC_KM
from utils.utils_torch import torch_interp_1d, torch_conv
from utils.set_root_paths import root_data_path


class KineticModel_2TC_curve_fit():

    """
    This is an auxiliary class which allows to run the curve fit on a specific patient.
    It is needed because PET_2TC_KM takes as input the IDIF (image-derived input function), but this is not a parameters which will not be optimized.
    For this reason, one KineticModel_2TC_curve_fit per patient is generated, with a specific IDIF.
    """

    def __init__(self, patient):
        self.patient = patient
    
    def read_idif(self, sample_time, t):
        idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", "DynamicFDG_" + self.patient + "_IDIF.txt")
        data = pd.read_csv(idif_txt_path, sep="\t")
        self.idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.idif_interp = torch_interp_1d(t, sample_time, self.idif)
        return self.idif_interp
    
    def PET_2TC_KM(self, t, k1, k2, k3, Vb):
        step = 0.1
        a = self.idif_interp
        e = (k2 + k3)*t
        b = k1 / (k2 + k3) * (k3 + k2*torch.exp(-e)) # 2TC irreversible
        c = torch_conv(a, b) * step
        PET = (1-Vb) * c + Vb * a
        return PET

if __name__ == '__main__':

    # Extract TAC per organ from test dataset
    # The organ-wise TAC are time consuming to extract, therefore they are saved in the image-derived folder
    test_patients_list = ["02"] #, "03", "08", "09", "20", "24", "25", "26", "28"]
    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")       # 
    t = torch.load(root_data_path + "/DynamicPET/t.pt")

    for patient in test_patients_list:
        if os.path.exists(os.path.join("image-derived", patient+"_TAC.npy")):
            print("FOUND:", os.path.join("image-derived", patient+"_TAC.npy"))
        else:
            label_map_path = glob.glob(root_data_path + "/DynamicPET/*DynamicFDG_"+str(patient)+"/NIFTY/Resampled/labels.nii.gz")[0]
            PET_list = glob.glob(root_data_path + "/DynamicPET/*DynamicFDG_"+str(patient)+"/NIFTY/Resampled/PET_*.nii.gz")
            save_path = glob.glob(root_data_path + "/DynamicPET/*DynamicFDG_" + str(patient))[0]

            if os.path.exists(label_map_path):
                label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
            else: 
                print("ERROR: missing label map", label_map_path)
                continue
            
            patient_dict = dict()           # This dict will save the organ-wise TAC
            for pet_path in PET_list:
                current_pet_ = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
                for label in np.unique(label_map_):
                    if not label in patient_dict.keys():
                        patient_dict[label] = list() 
                    current_organ = current_pet_[label_map_==label]
                    current_mean = np.mean(current_organ) / 1000        # from Bq/ml to kBq/ml
                    patient_dict[label].append(current_mean)
            np.save(os.path.join("image-derived", patient + "_TAC.npy"), patient_dict)

        # Visualize organ-wise TAC
        patient_dict = np.load(os.path.join("image-derived", patient + "_TAC.npy"), allow_pickle=True).item()
        fig = plt.figure()
        for k in patient_dict.keys():
            plt.plot(time_stamp, np.array(patient_dict[k]), label=labels_to_organ[k])
            plt.title("Patient: "+patient)
            plt.grid()
            plt.legend()
        plt.show()

    # Find the kinetic parameters per organ using the curve fit
    for patient in test_patients_list:
        patient_dict = np.load(os.path.join("image-derived", patient + "_TAC.npy"), allow_pickle=True).item()
        
        curve_fit_dict = dict()
        for k in patient_dict.keys():
            # Image derived TAC
            current_TAC = torch.from_numpy(np.array(patient_dict[k]))
            current_TAC_interp = torch_interp_1d(t, time_stamp, current_TAC)
            
            # Curve fit
            KM_2TC = KineticModel_2TC_curve_fit(patient)
            idif = KM_2TC.read_idif(time_stamp, t)
            p, pcov2 = curve_fit(KM_2TC.PET_2TC_KM, t, current_TAC_interp, p0=[0.1, 0.1, 0.01, 0.01], bounds=([0.01, 0.01, 0.01, 0], [2, 3, 1, 1]), diff_step=0.001)
            k1, k2, k3, Vb = p
            curve_fit_dict[k] = dict()
            curve_fit_dict[k]["k1"] = k1
            curve_fit_dict[k]["k2"] = k2
            curve_fit_dict[k]["k3"] = k3
            curve_fit_dict[k]["Vb"] = Vb
        fit_df = pd.DataFrame.from_dict(curve_fit_dict).T
        new_row_names = list()
        for r in fit_df.index:
            new_row_names.append(labels_to_organ[int(r)])
        fit_df.index = new_row_names
        fit_df.to_excel(os.path.join("image-derived", patient + "_curve_fit_params.xlsx"))

        # Visualize measured organ-wise TAC and the estimated one
        estimated_TAC_interp = PET_2TC_KM(idif, t, k1, k2, k3, Vb)

        fig = plt.figure()
        plt.plot(time_stamp, current_TAC, "-o", label=labels_to_organ[k])
        plt.plot(t, estimated_TAC_interp.detach().numpy(), "k")
        plt.ylim([0, 35])
        plt.xlim([0, 5])
        plt.grid()
        plt.legend()
        plt.show()
