import os, warnings
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset
from data_provider.timefeatures import time_features
from data_provider.T1DMS_Dataset import kalman_cgm
warnings.filterwarnings(action='ignore')

#####################################
#             DiaTrend              #
#####################################

class DiaTrend_Dataset(Dataset):
    """
    The DiaTrend dataset for Torch training.
    """
    def __init__(self, raw_df, seq_length, label_length, pred_length, external_mean=None, external_std=None, 
                       use_delta=False, gaussian: bool=False, Experiment_mode=None, sensor_sampling=5):
        
        raw_df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.sensor_sampling = sensor_sampling    
        self.seq_length = int(seq_length/self.sensor_sampling)
        self.label_length = int(label_length/self.sensor_sampling)
        self.pred_length = int(pred_length/self.sensor_sampling)
        self.use_delta = use_delta  # Whether to use delta-based prediction target
        self.gaussian = gaussian
        self.example_len = self.seq_length + self.pred_length
        self.mode = Experiment_mode
        self.data, self.data_stamp = self._initial(raw_df)
        self.example_indices = self._example_indices()
        self._standardise(external_mean, external_std)

        # post check
        for i in range(len(self)):
            if torch.isnan(self[i][5]).any():
                raise ValueError("NaN detected in dataset!")

    def _initial(self, raw_df):
        # Time processing
        if 'date' in raw_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(raw_df['date']):
                raw_df['date'] = pd.to_datetime(raw_df['date'])
        else:
             raise ValueError("Date column 'date' not found")
        df_stamp = raw_df[['date']].rename(columns={'date': 'index_new'})
        data_stamp = time_features(df_stamp, timeenc=1, freq='min')

        # Feature extraction
        glucose = raw_df['CGM'].to_numpy(dtype=np.float32)
        if 'CHO' in raw_df.columns:
            carbs = raw_df['CHO'].to_numpy(dtype=np.float32)
            
        if 'Basal' in raw_df.columns and 'Bolus' in raw_df.columns:
            # Conversion logic (calculated per 5-minute sampling window):
            # Bolus (U/5min): Already aggregated in 5-min window, convert to pmol by multiplying 6000.
            # Basal (U/5min): Preprocessed to be per 5-min window, convert to pmol by multiplying 6000.
            # Final insulin = (Basal + Bolus) * 6000, unit: pmol/5min
            insulin = (raw_df['Basal']*6000 + raw_df['Bolus']*6000).to_numpy(dtype=np.float32)

        if self.gaussian:
            glucose = kalman_cgm(glucose)

        # default order: Carbs, Insulin, Glucose
        if self.mode == "ggg":
            return np.array([
                glucose,
                glucose,
                glucose
            ], dtype=np.float32).T, data_stamp
        elif self.mode == "cgg":
            return np.array([
                carbs,
                glucose,
                glucose
            ], dtype=np.float32).T, data_stamp
        elif self.mode == "gig":
            return np.array([
                glucose,
                insulin,
                glucose
            ], dtype=np.float32).T, data_stamp
        else:
            # Default
            return np.array([
                carbs,
                insulin,
                glucose,
            ], dtype=np.float32).T, data_stamp

    def _example_indices(self):
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
        start_row, end_row = self.example_indices[idx]
        res = torch.from_numpy(self.data[start_row: end_row + 1, :])
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


def prepare_DiaTrend_data(root_dir, 
                          seq_length, 
                          label_length, 
                          pred_length, 
                          use_delta=False,
                          gaussian=False, 
                          Experiment_mode=None):

    DiaTrend_processed_dataset_dir = "Glucose_Data/DiaTrend_Dataset/Processed"
    train_patients = ["30", "36", "38", "39", "29", "31", "37", "45", "46", "50", "52"]
    validate_patients = ["47", "51"]
    test_patients = ["49", "54", "42", "53"]

    # Check existence
    all_patients = train_patients + validate_patients + test_patients
    need_preprocessing = False
    for p in all_patients:
        if not os.path.exists(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))):
             need_preprocessing = True
             break
    if need_preprocessing:
         print(f"Missing processed data. Re-running preprocessing...")
         preprocess_DiaTrend(root_dir)


    all_data = []
    for p in train_patients:
        train_df = pd.read_csv(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p)))
        all_data.append(train_df)
        # print("Global data :", test_df["glucose"])
    global_df = pd.concat(all_data, axis=0)
    scalar = DiaTrend_Dataset(global_df, seq_length, label_length, pred_length, use_delta=use_delta, gaussian=gaussian, Experiment_mode=Experiment_mode)

    global_train_set = torch.utils.data.ConcatDataset(
            DiaTrend_Dataset(
                raw_df=pd.read_csv(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))),
                seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian, 
                external_mean=scalar.mean,
                external_std=scalar.std,
                Experiment_mode=Experiment_mode
            ) for p in train_patients)

    global_validate_set = torch.utils.data.ConcatDataset(
            DiaTrend_Dataset(
                raw_df=pd.read_csv(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))),
                seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                external_mean=scalar.mean,
                external_std=scalar.std,
                Experiment_mode=Experiment_mode
            ) for p in validate_patients)

    global_test_set = torch.utils.data.ConcatDataset(
            DiaTrend_Dataset(
                raw_df=pd.read_csv(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))),
                seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                external_mean=scalar.mean,
                external_std=scalar.std,
                Experiment_mode=Experiment_mode
            ) for p in test_patients)

    return global_train_set, global_validate_set, global_test_set, test_patients


def prepare_DiaTrend_zeroshot_data(root_dir, 
                          seq_length, 
                          label_length, 
                          pred_length, 
                          external_mean,
                          external_std,
                          use_delta=False,
                          gaussian=False, 
                          Experiment_mode=None):

    DiaTrend_processed_dataset_dir = "Glucose_Data/DiaTrend_Dataset/Processed"
    test_patients = ["30", "36", "38", "39", "29", "31", "37", "45", "46", "50", "52", 
                     "47", "51",
                     "49", "54", "42", "53"]

    # Check existence
    need_preprocessing = False
    for p in test_patients:
        if not os.path.exists(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))):
             need_preprocessing = True
             break
    if need_preprocessing:
         print(f"Missing processed data. Re-running preprocessing...")
         preprocess_DiaTrend(root_dir)

    global_test_set = torch.utils.data.ConcatDataset(
            DiaTrend_Dataset(
                raw_df=pd.read_csv(os.path.join(root_dir, DiaTrend_processed_dataset_dir, "Processed_Subject{}.csv".format(p))),
                seq_length=seq_length, label_length=label_length, pred_length=pred_length, use_delta=use_delta, gaussian=gaussian,
                external_mean=external_mean,
                external_std=external_std,
                Experiment_mode=Experiment_mode
            ) for p in test_patients)

    return global_test_set, test_patients


def preprocess_DiaTrend(main_dir,  Use_IOB=False):
    # Set the directory where the Excel files are located
    directory = os.path.join(main_dir, "DiaTrend_Dataset/Raw")  # Adjust this to your files' directory

    # Names of the sheets we want to manipulate
    cgm_sheet_name = 'CGM'
    basal_sheet_name = 'Basal'
    bolus_sheet_name = 'Bolus'

    # We assume the date column in the sheets is named 'Date'
    date_column = 'date'
    rate_column = 'rate'  # Replace with the actual name of the rate column in Basal sheet
    cgm_column = 'mg/dl'

    # Process files
    selected_file_list = []
    cnt = 1
    for i, filename in enumerate(os.listdir(directory)):
        if filename.startswith("Subject") and filename.endswith(".xlsx") and filename != 'Subject_test.xlsx':
            file_path = os.path.join(directory, filename)
            xls = pd.ExcelFile(file_path)
            if (len(xls.sheet_names)==3):
                print(f"[{cnt}] Original File Name: {filename}"); cnt+=1;
                selected_file_list.append(filename)
                original_file_path = os.path.join(directory, filename)
                os.makedirs(directory.replace('Raw', 'Processed'), exist_ok=True)
                new_dir = directory.replace('Raw', 'Processed')
                new_filename = 'Processed_' + filename.replace('.xlsx', '.csv')
                
                new_file_path = os.path.join(new_dir, new_filename)

                '''
                Processing CGM
                '''
                # Load the 'CGM' sheet into a pandas DataFrame
                cgm_df = pd.read_excel(original_file_path, sheet_name=cgm_sheet_name)        
                # Convert the date column to datetime
                cgm_df[date_column] = pd.to_datetime(cgm_df[date_column])        
                # Sort the data based on the date column
                cgm_df.sort_values(by=date_column, inplace=True)  

                daily_counts = cgm_df[date_column].dt.floor('d').value_counts()        
                # Filter out days with fewer than 276 entries
                valid_days = daily_counts[daily_counts >= 276].index     
                
                # Round the 'DateTime' column to the nearest 5 minutes
                cgm_df[date_column] = cgm_df[date_column].dt.round('5min')
                # Set the date column as the index because resample works on the index
                cgm_df.set_index(date_column, inplace=True)
                # Resample the DataFrame to 5-minute intervals
                cgm_df = cgm_df.resample('5min').first()
                
                # if the cgm of 1 days has missings more than 20 minutes, remove that day's data
                not_missed_cgm_df = remove_consecutive_nans(cgm_df, cgm_column, 4, date_column)

                # indexing interpolated points
                not_missed_cgm_df['is_interpolated'] = False
                not_missed_cgm_df.loc[pd.isna(not_missed_cgm_df['mg/dl']), 'is_interpolated'] = True
                
                # linear interpolate data which has missings equal or less than 20 minutes
                interp_cgm_df = not_missed_cgm_df.copy()
                interp_cgm_df['mg/dl'] = not_missed_cgm_df['mg/dl'].interpolate(method='cubic')
                cgm_df = interp_cgm_df.drop(columns = ['group', 'nan_count']).reset_index(drop=True)

                cgm_df.loc[cgm_df['mg/dl']<40, 'mg/dl'] = 40
                cgm_df.loc[cgm_df['mg/dl']>400, 'mg/dl'] = 400

                '''
                Processing Insulin
                '''
                
                # Load the 'Basal' and 'Bolus' sheets
                basal_df = pd.read_excel(original_file_path, sheet_name=basal_sheet_name)
                bolus_df = pd.read_excel(original_file_path, sheet_name=bolus_sheet_name)
                
                # Convert the date columns to datetime
                basal_df[date_column] = pd.to_datetime(basal_df[date_column])        
                # Round the 'DateTime' column to the nearest 5 minutes
                basal_df[date_column] = basal_df[date_column].dt.round('5min')
                basal_df[rate_column] = basal_df[rate_column].fillna(0)  # Fill NaN values in 'Rate' column with 0
                # First, ensure your date column is in datetime format and sorted
                basal_df = basal_df.sort_values(by=date_column)
                # Set the date column as the index because resample works on the index
                basal_df.set_index(date_column, inplace=True)
        
                # Resample the DataFrame to 5-minute intervals
                basal_df = basal_df.resample('5min').first()
                basal_df['rate'] = basal_df['rate'].fillna(method='ffill')
                basal_df['rate'] = basal_df['rate'] / 12
                basal_df.reset_index(inplace=True)
        
                ### Bolus
                bolus_df[date_column] = pd.to_datetime(bolus_df[date_column])
                bolus_df[date_column] = bolus_df[date_column].dt.round('5min')
                bolus_df = bolus_df.sort_values(by=date_column)
                bolus_df.set_index(date_column, inplace=True)
        
                # Resample the DataFrame to 5-minute intervals
                bolus_df = bolus_df.resample('5min').first()
                bolus_df['normal'] = bolus_df['normal'].fillna(0)
                bolus_df['carbInput'] = bolus_df['carbInput'].fillna(0)
        
                # Reset the index so that date_column becomes a column again
                bolus_df.reset_index(inplace=True)

                combined_df = pd.merge(cgm_df, basal_df.loc[:, [date_column, 'rate']], on='date', how='left')
                combined_df = pd.merge(combined_df, bolus_df.loc[:, [date_column, 'normal', 'carbInput']], on='date', how='left')
                
                # Giving the columns appropriate names
                combined_df.columns = ['date', 'CGM', 'is_interpolated', 'Basal', 'Bolus', 'CHO']
                combined_df['INS'] = combined_df['Bolus'] + combined_df['Basal']
                
                # Reorder columns
                combined_df = combined_df[['date', 'CHO', 'Basal', 'Bolus', 'INS', 'CGM', 'is_interpolated']]
                combined_df.dropna(subset=['Basal', 'Bolus'], how='any', inplace=True)

                daily_counts = combined_df[date_column].dt.floor('d').value_counts()        
                valid_days = daily_counts[daily_counts >= 276].index
                valid_combined_df = filter_by_date(combined_df, valid_days, date_column)
                valid_combined_df = valid_combined_df.reset_index(drop=True)

                # Save the modified DataFrames to new csv file
                valid_combined_df.to_csv(new_file_path, index=False)
                print(f"--> [Saved] New File Name: {new_filename}")

    if Use_IOB:
        print('-'*50)
        print(f"[IOB Transformation & Combine to one .pt file and Save]", end = ' ')
        processed_dir = directory.replace('Raw', 'Processed')
        file_list = os.listdir(processed_dir)
        data_list = []
        for i, file in enumerate(file_list):
            if (os.path.splitext(file)[1]=='.csv'):
                _data = pd.read_csv(os.path.join(processed_dir, file))
                _data['IOB'] = get_iob(_data)
                _data['SID'] = file.split('.')[0].split('_')[1][-2:]        
                data_list.append(_data)
        print(f"{len(data_list)} Patients")

        final_fp = os.path.join(processed_dir, 'cgm+ins+iob.pt')
        print(f"Final Processed File Path: {final_fp}")
        torch.save(data_list, final_fp)


# Function to remove each day which has more than 20 min consecutive CGM missings
def remove_consecutive_nans(df, column, n, date_column):
    df = df.reset_index()
    df['group'] = (df[column].notna().cumsum())
    df['nan_count'] = df.groupby('group')[column].transform(lambda x: x.isna().sum())
    
    days_missing = df[(df.nan_count>=n+1) & (pd.isna(df[column]))][date_column].unique()
    df_cleaned = df[~df[date_column].isin(days_missing)]
    
    return df_cleaned

# Function to filter the DataFrame by valid dates
def filter_by_date(df, valid_dates, date_column):
    return df[df[date_column].dt.floor('d').isin(valid_dates)]

class IOB:
    def __init__(self):
        self.x1, self.x2 = 0, 0  
        self.dx1, self.dx2 = 0, 0 
        self.kdia = 0.025 

    def step(self, insulin):
        dx1 = insulin - self.kdia * self.x1
        dx2 = self.kdia * self.x1 - self.kdia * self.x2
        self.x1, self.x2 = self.x1 + dx1, self.x2 + dx2 # euler

    def get_IOB(self):
        return self.x1+self.x2


def get_iob(data):
    IOB_calculator = IOB()
    ins = data.INS 
    ins_1m, iob_1m = np.zeros([len(ins)*5]), np.zeros([len(ins)*5])
    
    ins_1m = [ins[i//5] if i%5==0 else 0 for i, _ in enumerate(ins_1m)] 
    for i, _ins in enumerate(ins_1m): # IOB 
        IOB_calculator.step(_ins)
        iob = IOB_calculator.get_IOB()
        iob_1m[i] = iob_1m[i] + iob

    iob_5m = iob_1m[::5]
    return iob_5m
