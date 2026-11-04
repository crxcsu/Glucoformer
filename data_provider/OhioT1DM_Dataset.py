import datetime
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from data_provider.timefeatures import time_features
from data_provider.T1DMS_Dataset import kalman_cgm

class OhioT1DM_Dataset(Dataset):
    """ 
    The OhioT1DM dataset for Torch training.
    """
    def __init__(self, raw_df, seq_length, label_length, pred_length, external_mean=None, external_std=None, 
                       use_delta=False, gaussian=False, Experiment_mode=None, sensor_sampling=5):
        """
        Args
            raw_df: dataframe
            example_len: int
            external_mean: [float]
                If none, self fit.
            external_std: [float]
                If none, self fit.
            unimodal: bool
                If True, data contains glucose only
        """
        raw_df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.sensor_sampling = sensor_sampling    
        self.seq_length = int(seq_length/self.sensor_sampling)
        self.label_length = int(label_length/self.sensor_sampling)
        self.pred_length = int(pred_length/self.sensor_sampling)
        self.use_delta = use_delta  # Whether to use delta-based prediction target
        self.gaussian = gaussian  # Boolean variable indicating whether Gaussian filtering is used 
        self.example_len = self.seq_length + self.pred_length
        self.mode = Experiment_mode
        self.data, self.data_stamp = self._initial(raw_df)  # (len, n_features)
        self.example_indices = self._example_indices()
        self._standardise(external_mean, external_std)
        # print("Dataset loaded, total examples: {}.".format(len(self)))

        # post check
        for i in range(len(self)):
            if torch.isnan(self[i][5]).any():# Index of the fourth position's return value in the __getitem__ function
                raise ValueError("NaN detected in dataset!")

    @staticmethod
    def str2dt(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def _initial(self, raw_df):
        times = [self.str2dt(s) for s in raw_df["index_new"]]
        glucose = raw_df["glucose"].to_numpy(dtype=np.float32)
        basal = raw_df["basal"].to_numpy(dtype=np.float32)
        bolus = raw_df["bolus"].to_numpy(dtype=np.float32)
        bolus_dur = raw_df["bolus_dur"].to_numpy(dtype=np.float32)
        carbs = raw_df["carbs"].to_numpy(dtype=np.float32)

        df_stamp = raw_df[['index_new']]
        # print(df_stamp)
        data_stamp = time_features(df_stamp, timeenc=1, freq='min')
        # print(data_stamp)

        bolus[np.isnan(bolus)] = 0.0
        carbs[np.isnan(carbs)] = 0.0

        total_len = len(times)

        # smooth out the long acting insulin
        i = 0
        while i < total_len:
            if bolus[i] > 0 and bolus_dur[i] > 0:
                # found a non-instant bolus
                j = 1
                while i + j < total_len:
                    if bolus[i + j] == bolus[i]:
                        j += 1
                    else:
                        break
                bolus[i: i + j] = bolus[i: i + j] / j
                i += j
            else:
                i += 1

        # Conversion logic (Calculated per 5-minute sampling window):
        # The dataset has been resampled to 5-minute intervals. 
        # The model needs the total amount of insulin (in pmol) administered within each 5-min window.
        # General conversion rate: 1 U = 6000 pmol.
        # 
        # Bolus (U): Directly converted to the total input amount in this 5-min window.
        # Contribution = bolus (U) * 6000 pmol/U
        #
        # Basal (U/h): Basal rates are given as continuous flow per hour (60 mins).
        # In a 5-minute window, the absolute volume flowed in = Basal(U/h) * (5 min / 60 min) = Basal * (1/12) U
        # Converted to pmol = Basal * (1/12) * 6000 pmol = Basal * 500 pmol
        injection = basal * 500 + bolus * 6000 #Aligned with 5-min aggregation, Unit: pmol/5min

        if self.gaussian:
            # glucose = gaussian_filter1d(glucose, sigma=1)
            # glucose = causal_gaussian_filter(glucose, sigma=1)
            glucose = kalman_cgm(glucose)

        # if self.mode == "ggg":
        #     return np.array([
        #         glucose,
        #         glucose,
        #         glucose
        #     ], dtype=np.float32).T, data_stamp
        # elif self.mode == "cgg":
        #     return np.array([
        #         glucose,
        #         carbs,
        #         glucose
        #     ], dtype=np.float32).T, data_stamp
        # elif self.mode == "gig":
        #     return np.array([
        #         glucose,
        #         injection,
        #         glucose
        #     ], dtype=np.float32).T, data_stamp
        # else:
        return np.array([
            carbs,
            injection,
            glucose,
        ], dtype=np.float32).T, data_stamp

    def _example_indices(self):
        """ Extract every possible example from the dataset, st. all data entry in this example is not missing.

        Returns:
            [(start_row, end_row)]
                Starting and ending indices for each possible example from this dataframe.
        """
        res = []
        total_len = self.data.shape[0]

        def look_ahead(start):
            end = start
            res = []
            while end < total_len:
                if np.any(np.isnan(self.data[end, :])):
                    break
                if end - start + 1 >= self.example_len:
                    res.append((end - self.example_len + 1, end))
                end += 1
            return res, end

        i = 0
        while i < total_len:
            if not np.any(np.isnan(self.data[i, :])):
                temp_res, temp_end = look_ahead(i)
                res += temp_res
                i = temp_end + 1
            else:
                i += 1
        return res

    def _standardise(self, external_mean=None, external_std=None):
        if external_mean is None and external_std is None:
            mean = []
            std = []
            for i in range(self.data.shape[1]):
                mean.append(np.nanmean(self.data[:, i]))
                std.append(np.nanstd(self.data[:, i]))
        else:
            mean = external_mean
            std = external_std
        self.mean = mean
        self.std = std
        for i in range(self.data.shape[1]):
            safe_std_val = self.std[i] if self.std[i] > 1e-9 else 1.0
            self.data[:, i] = (self.data[:, i] - mean[i]) / safe_std_val

    def inverse_transform(self, data, feature_indices=None):
        """
        Inverse transform data.
        If data has same number of features as mean/std (3), restores all.
        If data is single feature (e.g. target), you must specify which feature index.
        For example, glucose data is typically at index 2.
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
             # Default to assuming it's the target (index 2: glucose)
             mean = self.mean[2]
             std = self.std[2]
        else:
             raise ValueError(f"Data shape {data.shape} does not match expected features. Please specify feature_indices.")
             
        # Use broadcasting
        return (data * std) + mean

    def __len__(self):

        return len(self.example_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx: int
        Returns:
            (example_len, channels)
        """
        start_row, end_row = self.example_indices[idx]
        res = torch.from_numpy(self.data[start_row: end_row + 1, :])

        if self.mode == "ggg":
            res[:, 0] = -1.0
            res[:, 1] = -1.0
        elif self.mode == "cgg":
            res[:, 1] = -1.0
        elif self.mode == "gig":
            res[:, 0] = -1.0

        encoder_input = res[:self.seq_length, :]
        decoder_input = res[self.seq_length-self.label_length: , :].clone()
        decoder_input[self.label_length:, :] = 0

        if self.use_delta:
            bg_current = self.data[start_row + self.seq_length - 1, 2] * self.std[2] + self.mean[2]
            bg_future = self.data[start_row + self.seq_length : end_row + 1, 2] * self.std[2] + self.mean[2]
            delta = bg_future - bg_current
            delta_scaler = delta / self.std[2]
            tgt = torch.tensor(delta_scaler, dtype=torch.float32).unsqueeze(-1)
        else:
            tgt = res[self.seq_length: , 2].unsqueeze(-1)

        mark = torch.from_numpy(self.data_stamp[start_row: end_row + 1, :])
        encoder_input_mark = mark[:self.seq_length, :].to(dtype=torch.float32)
        decoder_input_mark = mark[self.seq_length-self.label_length: , :].to(dtype=torch.float32)

        return encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt,  res


def prepare_Ohio_data(root_dir, 
                      seq_length, 
                      label_length, 
                      pred_length,
                      use_delta=False, 
                      gaussian=False, 
                      Experiment_mode=None):

    ohio_processed_dataset_dir = "Glucose_Data/OhioT1DM_Dataset/OhioT1DM_processed_dataset"
    train_patients = ["540", "544", "559", "567", "570", "575", "584", "591"]
    validate_patients = ["588", "552"]
    test_patients = ["563", "596"]
    
    all_data = []
    for p in train_patients:
        train_df = pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_train.csv".format(p)))
        all_data.append(train_df)
        test_df = pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_test.csv".format(p)))
        all_data.append(test_df)
        # print("Global data :", test_df["glucose"])
    global_df = pd.concat(all_data, axis=0)
    scalar = OhioT1DM_Dataset(global_df, seq_length, label_length, pred_length, use_delta=use_delta, gaussian=gaussian, Experiment_mode=Experiment_mode)

    global_train_set = torch.utils.data.ConcatDataset(
        sum(zip(
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_train.csv".format(p))), 
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in train_patients
                ],
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_test.csv".format(p))),  
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in train_patients
                ]
            ), ())
        )

    global_validate_set = torch.utils.data.ConcatDataset(
        sum(zip(
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_train.csv".format(p))), 
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in validate_patients
                ],
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_test.csv".format(p))),  
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in validate_patients
                ]
            ), ())
        )

    global_test_set = torch.utils.data.ConcatDataset(
        sum(zip(
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_train.csv".format(p))), 
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in test_patients
                ],
                [
                    OhioT1DM_Dataset(
                        raw_df=pd.read_csv(os.path.join(root_dir, ohio_processed_dataset_dir, "{}_test.csv".format(p))),  
                        seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                        external_mean=scalar.mean,
                        external_std=scalar.std,
                        Experiment_mode=Experiment_mode
                    ) for p in test_patients
                ]
            ), ())
        )

    return global_train_set, global_validate_set, global_test_set, test_patients, scalar


def prepare_Ohio_LOSO(root_dir,
                      seq_length,
                      label_length,
                      pred_length,
                      use_delta=False,
                      gaussian=False,
                      Experiment_mode=None,
                      n_val=2,
                      patient_ring=None):
    """
    Generate 12 LOSO folds, return list[dict].
    Each fold contains:
        fold / test_patient / train_patients / val_patients /
        train_set / val_set / test_set / scaler

    Cyclic round-robin rule:
        For fold i:
            test = ring[i]
            pool = next 11 patients on the ring after the test patient (cyclic order)
            val  = pool[:n_val]      # first n_val patients adjacent to test, sliding per fold
            train = pool[n_val:]     # remaining patients

    Normalization statistics are fitted ONLY on training patients of this fold
    to avoid validation/test leakage.

    Usage:
        folds = prepare_Ohio_LOSO(root_dir, seq_length=..., label_length=..., pred_length=...)
        for f in folds:
            model = train(f["train_set"], val_set=f["val_set"])  # select best epoch using val
            rmse  = evaluate(model, f["test_set"])               # evaluate on held-out patient
    """
    import os
    import pandas as pd
    import torch

    OHIO_PROCESSED_DIR = "Glucose_Data/OhioT1DM_Dataset/OhioT1DM_processed_dataset"
    ring = list(patient_ring) if patient_ring else [
        "540", "544", "559", "567", "570", "575",
        "584", "591", "588", "552", "563", "596",
    ]
    N = len(ring)
    if n_val < 1 or n_val >= N - 1:
        raise ValueError(f"n_val must be between 1 and {N - 2}, inclusive (got {n_val})")

    folds = []
    for i in range(N):
        test_p = ring[i]
        pool = [ring[(i + 1 + j) % N] for j in range(N - 1)] 
        val_p = pool[:n_val]
        train_p = pool[n_val:]

        # Only use training to fit the normalized statistics (for leakage prevention)
        all_train_df = pd.concat(
            [pd.read_csv(os.path.join(root_dir, OHIO_PROCESSED_DIR, f"{p}_{sp}.csv"))
             for p in train_p for sp in ("train", "test")],
            axis=0,
        )
        scaler = OhioT1DM_Dataset(
            all_train_df, seq_length, label_length, pred_length,
            use_delta=use_delta, gaussian=gaussian, Experiment_mode=Experiment_mode,
        )

        sets = {}
        for key, plist in (("train_set", train_p),
                           ("val_set", val_p),
                           ("test_set", [test_p])):
            sets[key] = torch.utils.data.ConcatDataset([
                OhioT1DM_Dataset(
                    raw_df=pd.read_csv(os.path.join(root_dir, OHIO_PROCESSED_DIR, f"{p}_{sp}.csv")),
                    seq_length=seq_length, label_length=label_length, pred_length=pred_length,
                    use_delta=use_delta, gaussian=gaussian,
                    external_mean=scaler.mean, external_std=scaler.std,
                    Experiment_mode=Experiment_mode,
                )
                for p in plist for sp in ("train", "test")
            ])

        folds.append({
            "fold": i,
            "test_patient": test_p,
            "train_patients": train_p,
            "val_patients": val_p,
            "train_set": sets["train_set"],
            "val_set": sets["val_set"],
            "test_set": sets["test_set"],
            "scaler": scaler,
        })
    return folds


def prepare_Ohio_all_subjects(root_dir,
                              seq_length,
                              label_length,
                              pred_length,
                              use_delta=False,
                              gaussian=False,
                              Experiment_mode=None):

    ohio_processed_dataset_dir = "Glucose_Data/OhioT1DM_Dataset/OhioT1DM_processed_dataset"
    all_patients = ["540", "544", "559", "567", "570", "575",
                    "584", "591", "588", "552", "563", "596"]

    all_data = []
    for p in all_patients:
        for sp in ("train", "test"):
            all_data.append(pd.read_csv(
                os.path.join(root_dir, ohio_processed_dataset_dir, "{}_{}.csv".format(p, sp))))
    global_df = pd.concat(all_data, axis=0)
    scalar = OhioT1DM_Dataset(global_df, seq_length, label_length, pred_length,
                              use_delta=use_delta, gaussian=gaussian, Experiment_mode=Experiment_mode)

    sub_datasets = []
    for p in all_patients:
        for sp in ("train", "test"):
            sub_datasets.append(OhioT1DM_Dataset(
                raw_df=pd.read_csv(
                    os.path.join(root_dir, ohio_processed_dataset_dir, "{}_{}.csv".format(p, sp))),
                seq_length=seq_length, label_length=label_length, pred_length=pred_length,
                use_delta=use_delta, gaussian=gaussian,
                external_mean=scalar.mean, external_std=scalar.std,
                Experiment_mode=Experiment_mode))
    global_set = torch.utils.data.ConcatDataset(sub_datasets)
    return global_set, all_patients, scalar