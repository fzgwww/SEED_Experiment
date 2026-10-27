import numpy as np
import os
import pickle

eeg_train_data =None
eeg_test_data = None
eye_train_data =None
eye_test_data = None
train_label = None
test_label = None

for i in range(1,15):
    if i==6 or i==7:
        continue
    for j in range(1,4):    
        eye_file_path = r"D:\Python\BCI\SEED\eye_tracking_feature\\" +str(i) + "_" + str(j)
        eeg_file_path = r"D:\Python\BCI\SEED\eeg_de_feature\eeg_used_4s\\"+str(i)+"_"+str(j)+".npz"
        eye_data = np.load(eye_file_path,allow_pickle=True)
        eeg_data = np.load(eeg_file_path,allow_pickle=True)

        eeg_train = None
        for k in pickle.loads(eeg_data["train_data"]).values():
            eeg_train = np.concatenate((eeg_train,k),axis=1) if eeg_train is not None else k
        eeg_test = None
        for k in pickle.loads(eeg_data["test_data"]).values():
            eeg_test = np.concatenate((eeg_test,k),axis=1) if eeg_test is not None else k
        
        eeg_train_data = np.concatenate((eeg_train_data,eeg_train),axis=0) if eeg_train_data is not None else eeg_train
        eeg_test_data = np.concatenate((eeg_test_data,eeg_test),axis=0) if eeg_test_data is not None else eeg_test
        eye_train_data = np.concatenate((eye_train_data,eye_data["train_data_eye"]),axis=0) if eye_train_data is not None else eye_data["train_data_eye"]
        eye_test_data = np.concatenate((eye_test_data,eye_data["test_data_eye"]),axis=0) if eye_test_data is not None else eye_data["test_data_eye"]
        train_label = np.concatenate((train_label,eeg_data["train_label"]),axis=0) if train_label is not None else eeg_data["train_label"]
        test_label = np.concatenate((test_label,eeg_data["test_label"]),axis=0) if test_label is not None else eeg_data["test_label"]
print(f"EEG训练集: {eeg_train_data.shape}, 眼动训练集: {eye_train_data.shape}, 真实标签: {train_label.shape}")