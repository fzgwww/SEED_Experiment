import numpy as np

eye_file_path = r"D:\Python\BCI\SEED\eye_tracking_feature\1_1"
eeg_file_path = r"D:\Python\BCI\SEED\eeg_de_feature\eeg_used_4s\1_1.npz"
eeg_file_path_1s = r"D:\Python\BCI\SEED\eeg_de_feature\eeg_used_1s\1_1.npz"

eye_data = np.load(eye_file_path, allow_pickle=True)
eeg_data = np.load(eeg_file_path, allow_pickle=True)
eeg_data_1s = np.load(eeg_file_path_1s, allow_pickle=True)

train_label = eeg_data["train_label"]
test_label = eeg_data["test_label"]
eeg_train_data = eeg_data["train_data"]
eeg_test_data = eeg_data["test_data"]
eye_train_data = eye_data["train_data_eye"]
eye_test_data = eye_data["test_data_eye"]

print(eeg_data_1s.keys())
print(eeg_data_1s["train_data"].shape, eeg_data_1s["test_data"].shape, eeg_data_1s["train_label"].shape, eeg_data_1s["test_label"].shape)
# datatype = eeg_train_data.dtype
# print(eeg_train_data.tobytes().decode('utf-8', errors='ignore')[:500])
#print(f"眼动训练集: {eye_train_data.shape},EEG训练集: {eeg_train_data.item().shape}, 真实标签: {train_label.shape}")