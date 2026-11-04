import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from data_provider.timefeatures import time_features, generate_random_start_dates
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter
from filterpy.kalman import KalmanFilter


def kalman_cgm(x, q=None, r=None, burn_in=30):
    """Standard causal Kalman denoising for CGM.
    x : 1-D CGM sequence (mg/dL), e.g. shape (865,)
    q : process noise variance lambda^2 (estimated adaptively if None)
    r : measurement noise variance sigma^2 (estimated adaptively if None)
    """
    x = np.asarray(x, float)
    if r is None or q is None:
        r, q = _estimate_qr(x, burn_in)        
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[x[0]]])                  # initial state
    kf.F = np.array([[1.0]])                   # state transition: random walk (gradual glucose drift)
    kf.H = np.array([[1.0]])                   # observation: measurement = truth + sensor noise
    kf.P *= 1.0                                # initial error covariance
    kf.Q = np.array([[q]])                     # process noise variance (how fast true glucose may move)
    kf.R = np.array([[r]])                     # measurement noise variance (CGM sensor noise level)
    return kf.batch_filter(x)[0][:, 0, 0]      # filtered state estimates, reshaped to (N,)

def _estimate_qr(x, burn_in=30):
    seg = x[:burn_in]
    q = max(np.var(np.diff(seg)), 1e-6)            # lambda^2 ~ variance of adjacent-step glucose change
    r = max(np.var(seg - seg.mean()) - q, 1e-6)    # sigma^2 ~ total variance - process variance
    return r, q


class T1DMS_Dataset(Dataset):
    def __init__(self, root_dir, seq_length, label_length, pred_length, dataset_type='T1DMS', external_mean=None, external_std=None,
                       use_delta=False, sensor_sampling=5, stride=1, gaussian: bool=True, Experiment_mode=None):
        """ 
        Set time steps for sample generation
        Encoder input:
            Take historical seq_length time steps: [0, 1, ..., 19]
        Decoder input:
            Use the last label_length time steps: [10, 11, ..., 19]
        Prediction target:
            Predict the future pred_length time steps: [20, 21, ..., 24]
        """
        if external_mean is None and external_std is None:
            sim_result_mat_file_path = os.path.join(root_dir, "Glucose_Data", f"{dataset_type}_Dataset", 'sim_results_train_dataset.mat')
            sim_data_mat_file_path = os.path.join(root_dir, "Glucose_Data", f"{dataset_type}_Dataset", 'sim_data_train_dataset.mat')
        else:
            sim_result_mat_file_path = os.path.join(root_dir, "Glucose_Data", f"{dataset_type}_Dataset", 'sim_results_validate_dataset.mat')
            sim_data_mat_file_path = os.path.join(root_dir, "Glucose_Data", f"{dataset_type}_Dataset", 'sim_data_validate_dataset.mat')
        self.sim_result_mat_data = scipy.io.loadmat(sim_result_mat_file_path)
        self.sim_data_mat_data = scipy.io.loadmat(sim_data_mat_file_path)
        self.gaussian = gaussian  # Boolean variable, indicates whether to use Gaussian filtering
        self.mode = Experiment_mode
        self.use_delta = use_delta  # Whether to use delta-based prediction target
        self.sensor_sampling = sensor_sampling    
        self.seq_length = int(seq_length/self.sensor_sampling)
        self.label_length = int(label_length/self.sensor_sampling)
        self.pred_length = int(pred_length/self.sensor_sampling)     
        self.stride = stride 
        self.mean = external_mean
        self.std = external_std
        self.data_matrix, self.data_matrix_scaler, self.data_stamps = self.load_data()

    def inverse_transform(self, data, feature_indices=None):
        """
        Inverse transform data.
        If data has same number of features as mean/std (5), restores all.
        If data is single feature (e.g. target), you must specify which feature index from the 5 scaled features it corresponds to.
        For example, sensor data is typically at index 2 within the 5 scaled features.
        """
        if self.mean is None or self.std is None:
             raise ValueError("Mean and Std not initialized.")
             
        if feature_indices is not None:
            mean = self.mean[feature_indices]
            std = self.std[feature_indices]
        elif data.shape[-1] == len(self.mean):
            mean = self.mean
            std = self.std
        elif data.shape[-1] == 1:
             # Default to assuming it's the target (index 2 out of 5) if shape matches 1 feature
             mean = self.mean[2]
             std = self.std[2]
        else:
             raise ValueError(f"Data shape {data.shape} does not match expected features usage. Please specify feature_indices.")
             
        # Use broadcasting
        return (data * std) + mean

    def load_data(self):
        """Construct dataset matrix"""
        # Get number of scenarios, subjects, and simulation length
        scenarios_num = self.sim_result_mat_data['data'].shape[1]
        subjects_num = self.sim_result_mat_data['data']['results'][0, 0].shape[1]

        indicators = ['CHO', 'injection', 'sensor']
            
        features_num = len(indicators)+2
        sim_len = self.sim_result_mat_data['data']['results'][0, 0]['time'][0, 0]['signals'][0, 0]['values'][0, 0].shape[0]    
        data_matrix = np.zeros((scenarios_num, subjects_num, features_num, int(((sim_len-1)/self.sensor_sampling)+1))) # Create empty matrix to store extracted data

        # Create arrays for subject age and weight
        # age_data, weight_data = [], []
        # for sub in range(subjects_num):
        #     age_data.append(self.sim_data_mat_data['Lstruttura']['AGE'][0,sub])
        #     weight_data.append(self.sim_data_mat_data['Lstruttura']['BW'][0,sub])
        # age_data = np.array(age_data).flatten()
        # weight_data = np.array(weight_data).flatten()

        time = self.sim_result_mat_data['data']['results'][0, 0]['time'][0, 0]['signals'][0, 0]['values'][0, 0].flatten()
        start_dates = generate_random_start_dates(scenarios_num) # is a list
        minutes_to_add = np.arange(0, len(time), self.sensor_sampling)
        # Iterate over each scenario
        all_data_stamp = []
        for start_date in start_dates:
            date_time_series = pd.to_datetime(start_date) + pd.to_timedelta(minutes_to_add, unit='m')
            df_stamp = pd.DataFrame(date_time_series, columns=['index_new'])
            data_stamp = time_features(df_stamp, timeenc=1, freq='min')
            all_data_stamp.append(data_stamp)
        data_stamps = np.vstack(all_data_stamp).reshape(scenarios_num, -1, data_stamp.shape[-1])

        # Iterate over each scenario and extract data
        for scn in range(scenarios_num):
            for sub in range(subjects_num):
                # Fill age feature
                # data_matrix[scn, sub, features_num-2, :] = np.tile(age_data[sub], int(((sim_len-1)/self.sensor_sampling)+1))
                # Fill weight feature
                # data_matrix[scn, sub, features_num-1, :] = np.tile(weight_data[sub], int(((sim_len-1)/self.sensor_sampling)+1))
                for feature, indicator in enumerate(indicators):
                    data = self.sim_result_mat_data['data']['results'][0, scn][indicator][0, sub]['signals'][0, 0]['values'][0, 0].flatten()
                    data_blocks = data[:sim_len-1].reshape(-1, self.sensor_sampling)
                    if indicator == 'sensor':
                        data_reshape = data_blocks.mean(axis=1)
                    elif indicator == 'CHO' or indicator == 'injection':
                        # For carbohydrate intake (CHO) and insulin (injection), the amounts consumed or administered within a 5-minute period should be summed up.
                        data_reshape = data_blocks.sum(axis=1)
                    data_reshape = np.insert(data_reshape, int(((sim_len-1)/self.sensor_sampling)), data[sim_len-1])
                    if self.gaussian and indicator == 'sensor':
                        data_reshape = kalman_cgm(data_reshape)
                        
                    data_matrix[scn, sub, feature, :] = data_reshape
        
        # Compute mean and std for each feature if not provided
        if self.mean is None and self.std is None:
            self.mean = np.nanmean(data_matrix, axis=(0, 1, 3))
            self.std = np.nanstd(data_matrix, axis=(0, 1, 3))
        safe_std = np.where(self.std > 1e-9, self.std, 1.0)
        # Broadcast to normalize 4D matrix (Scenarios, Subjects, Features, Time)
        data_matrix_scaler = (data_matrix - self.mean[None, None, :, None]) / safe_std[None, None, :, None]

        self.indices = [] 
        for scn in range(scenarios_num):
            for sub in range(subjects_num):
                for i in range(0, int(((sim_len-1)/self.sensor_sampling)+1)-self.seq_length-self.pred_length+1, self.stride):
                    self.indices.append((scn, sub, i))

        return data_matrix[:,:,:3,:], data_matrix_scaler[:,:,:3,:], data_stamps

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):
        scn, sub, i = self.indices[idx]
        encoder_input_scaler = self.data_matrix_scaler[scn, sub, :, i : i+self.seq_length].transpose(1, 0)
        decoder_input_scaler = self.data_matrix_scaler[scn, sub, :, i+self.seq_length-self.label_length : i+self.seq_length+self.pred_length].transpose(1, 0)
        encoder_input_scaler_sample = torch.tensor(encoder_input_scaler, dtype=torch.float32)
        decoder_input_scaler_sample = torch.tensor(decoder_input_scaler, dtype=torch.float32)

        if self.mode == "ggg":
            encoder_input_scaler_sample[:, 0] = -1.0
            encoder_input_scaler_sample[:, 1] = -1.0
            decoder_input_scaler_sample[:, 0] = -1.0
            decoder_input_scaler_sample[:, 1] = -1.0
        elif self.mode == "cgg":
            encoder_input_scaler_sample[:, 1] = -1.0
            decoder_input_scaler_sample[:, 1] = -1.0
        elif self.mode == "gig":
            encoder_input_scaler_sample[:, 0] = -1.0
            decoder_input_scaler_sample[:, 0] = -1.0
        # print(f"encoder_input_scaler_sample before zeroing: {encoder_input_scaler_sample[0:9,:]}")
        # print(f"decoder_input_scaler_sample before zeroing: {decoder_input_scaler_sample[0:9,:]}")

        decoder_input_scaler_sample[self.label_length : self.label_length+self.pred_length, :] = 0

        if self.use_delta:
            bg_current = self.data_matrix[scn, sub, -1, i+self.seq_length-1]
            bg_future = self.data_matrix[scn, sub, -1, i+self.seq_length : i+self.seq_length+self.pred_length]
            delta = bg_future - bg_current
            # print(f"bg_current: {bg_current}, bg_future: {bg_future}, delta: {delta}")
            delta_scaler = delta / self.std[2]  
            tgt_scaler = delta_scaler
        else:
            tgt_scaler = self.data_matrix_scaler[scn, sub, -1 , i+self.seq_length : i+self.seq_length+self.pred_length] 
        tgt_scaler_sample = torch.tensor(tgt_scaler, dtype=torch.float32).unsqueeze(-1)

        mark = torch.from_numpy(self.data_stamps)
        encoder_input_mark = mark[scn, i : i+self.seq_length,:].to(dtype=torch.float32)
        decoder_input_mark = mark[scn, i+self.seq_length-self.label_length : i+self.seq_length+self.pred_length, :].to(dtype=torch.float32)

        return encoder_input_scaler_sample, encoder_input_mark, decoder_input_scaler_sample, decoder_input_mark, tgt_scaler_sample, torch.zeros(1)


def prepare_T1DMS_data(dataset_str, root_dir, seq_length, label_length, pred_length, use_delta, sensor_sampling, gaussian=True):
    dataset_types = [d.strip() for d in dataset_str.split(',')]
    train_datasets = []
    validate_datasets = []
    
    # Step 1: Load all train datasets to get their individual means and stds
    for d_type in dataset_types:
        t_ds = T1DMS_Dataset(
                       dataset_type=d_type, root_dir=root_dir, seq_length=seq_length, label_length=label_length, pred_length=pred_length,
                        use_delta=use_delta, sensor_sampling=sensor_sampling, gaussian=gaussian)
        train_datasets.append(t_ds)
    
    # Step 2: Compute exact global mean and std
    total_elements = 0
    global_sum = 0
    global_sum_sq = 0
    for t_ds in train_datasets:
        n = t_ds.data_matrix.shape[0] * t_ds.data_matrix.shape[1] * t_ds.data_matrix.shape[3]
        total_elements += n
        global_sum += n * t_ds.mean
        global_sum_sq += n * (t_ds.std**2 + t_ds.mean**2)
        
    global_mean = global_sum / total_elements
    global_variance = (global_sum_sq / total_elements) - global_mean**2
    global_std = np.sqrt(np.maximum(global_variance, 0))
    
    # Step 3: Re-scale train datasets and create validate datasets with global moments
    for t_ds, d_type in zip(train_datasets, dataset_types):
        t_ds.mean = global_mean
        t_ds.std = global_std
        
        safe_std = np.where(global_std[:3] > 1e-9, global_std[:3], 1.0)
        t_ds.data_matrix_scaler = (t_ds.data_matrix - global_mean[:3][None, None, :, None]) / safe_std[None, None, :, None]
        
        v_ds = T1DMS_Dataset(
                        dataset_type=d_type, root_dir=root_dir, seq_length=seq_length, label_length=label_length, pred_length=pred_length,
                        external_mean=global_mean, external_std=global_std, 
                        use_delta=use_delta, sensor_sampling=sensor_sampling, gaussian=gaussian)
        validate_datasets.append(v_ds)
    
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        validate_dataset = ConcatDataset(validate_datasets)
    else:
        train_dataset = train_datasets[0]
        validate_dataset = validate_datasets[0]
        
    return train_dataset, validate_dataset, None