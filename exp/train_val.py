import warnings, os
import json
import glob
import datetime
import math
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from exp.inference import set_model_parameters, run_inference
from data_provider.T1DMS_Dataset import DataLoader as T1DMS_DataLoader, prepare_T1DMS_data
from data_provider.OhioT1DM_Dataset import prepare_Ohio_data
from data_provider.DiaTrend_Dataset import prepare_DiaTrend_data, prepare_DiaTrend_zeroshot_data
from exp.metrics import MultiHorizonMSELoss
from utils.tools import *


def main(config, main_path=os.getcwd()):
    has_test = False

    for seed in range(0, config.repeat_times):
        config.seed = seed
        set_seed(config.seed)

        config, model, device = set_model_parameters(config)

        if "T1DMS" in config.dataset:
            config.pretrain_model_pth = None

        if seed == 0:
            if config.experiment_dir is None:
                time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                folder_id = '{}_{}_pretrain{}_bs{}_lr{:.1e}_dm{}_el{}_{}'.format(
                            config.dataset,
                            config.model_name,
                            1 if (config.pretrain_model_pth is not None and config.pretrain_model_pth != 'None') else 0,
                            config.batch_size,
                            config.lr,
                            getattr(config, 'd_model', 0),
                            getattr(config, 'e_layers', 0),
                            time_stamp
                        )
                config.experiment_dir = os.path.join("results", folder_id)
                LOG_DIR =  os.path.join(main_path, config.experiment_dir)
                setup_folders(LOG_DIR, config)

                with open(os.path.join(LOG_DIR, f"config_{time_stamp}.json"), 'w') as f:
                    json.dump(vars(config), f, indent=4)
            else:
                LOG_DIR = os.path.join(main_path, config.experiment_dir)
                setup_folders(LOG_DIR, config, clear=True)
                with open(os.path.join(LOG_DIR, f"config.json"), 'w') as f:
                    json.dump(vars(config), f, indent=4)

            print("\n" + "="*30 + " Configuration " + "="*30)
            print(json.dumps(vars(config), indent=4))
            print("="*75 + "\n")

        if "T1DMS" in config.dataset:
            train_dataset, validate_dataset, test_dataset = prepare_T1DMS_data(
                dataset_str=config.dataset, root_dir=main_path,
                seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len,
                use_delta=config.use_delta, sensor_sampling=config.sensor_sampling, gaussian=True)
        elif config.dataset == "OhioT1DM":
            train_dataset, validate_dataset, test_dataset, test_patients, scalar = prepare_Ohio_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, use_delta=config.use_delta)
            zeroshot_dataset, DiaTrend_zeroshot_patients = prepare_DiaTrend_zeroshot_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len,
                external_mean=scalar.mean, external_std=scalar.std, use_delta=config.use_delta)
        elif config.dataset == "DiaTrend":
            train_dataset, validate_dataset, test_dataset, test_patients = prepare_DiaTrend_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, use_delta=config.use_delta)

        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        train(config, LOG_DIR, train_dataloader, validate_dataloader, device, model)

        if test_dataset is not None:
            has_test = True

    if has_test:
        print(f"\n>>> 调用 inference.py 执行训练后推理 "
              f"(config_dir={LOG_DIR}, root={main_path})")
        run_inference(LOG_DIR, main_path=main_path)


def train(config, LOG_DIR, train_dataloader, validate_dataloader, device, model, crossval=False):
    validate_min_loss = float('inf')
    early_stop_cnt = 0
    best_state = None

    # normal runs persist the best weights / loss log to disk; CV keeps the best
    # weights in memory only (the caller evaluates the model right after training)
    if not crossval:
        model_path = os.path.join(LOG_DIR + config.model_save_path, f'{config.model_name}_seed{config.seed}.pth')
        loss_record_csv_path = os.path.join(LOG_DIR + config.loss_record_path, f'loss_record_{config.seed}.csv')
        fig_save_path = os.path.join(LOG_DIR + config.loss_record_path, f'plot_loss_record_{config.seed}.png')

    if config.pretrain_model_pth is not None and config.pretrain_model_pth != 'None':
        model.load_state_dict(torch.load(config.pretrain_model_pth))
        print("Loaded pre-trained model weights successfully!\n")

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    if "T1DMS" in config.dataset:
        criterion3 = MultiHorizonMSELoss(pred_len=int(config.pred_len / config.sensor_sampling), mode='power', beta=2)
    else:
        criterion3 = MultiHorizonMSELoss(pred_len=int(config.pred_len / config.sensor_sampling), mode='power', beta=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.pre_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=config.pre_gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs * len(train_dataloader))

    for epoch in range(1, config.train_epochs + 1):
        total_train_MSE_loss = 0
        total_train_MAE_loss = 0
        total_validate_MSE_loss = 0
        total_validate_MAE_loss = 0

        print(f"{config.model_name} model is training... Epoch: {epoch}/{config.train_epochs}")
        current_time = datetime.datetime.now()
        model.train()
        pbar = tqdm(train_dataloader)
        for _, (encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _) in enumerate(pbar):
            encoder_input = encoder_input.to(device)
            encoder_input_mark = encoder_input_mark.to(device)
            decoder_input = decoder_input.to(device)
            decoder_input_mark = decoder_input_mark.to(device)
            tgt = tgt.to(device)
            last_lr = optimizer.param_groups[0]['lr']

            optimizer.zero_grad()
            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            train_dataset = train_dataloader.dataset
            if hasattr(train_dataset, "datasets"):
                inverse_transform_func = train_dataset.datasets[0].inverse_transform
                mean = train_dataset.datasets[0].mean[2]
                std = train_dataset.datasets[0].std[2]
            else:
                inverse_transform_func = train_dataset.inverse_transform
                mean = train_dataset.mean[2]
                std = train_dataset.std[2]

            if config.use_delta:
                # Compute delta-based and absolute BG prediction losses and combine them for training
                bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                mean = torch.tensor(mean, device=output.device)
                std = torch.tensor(std, device=output.device)
                bg_current_real = bg_current_norm * std + mean
                delta_pred_real = output * std
                delta_true_real = tgt * std
                bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
                bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
                train_MSE_loss_inverse = criterion1(bg_pred_real, bg_true_real)
                train_MAE_loss_inverse = criterion2(bg_pred_real, bg_true_real)
                if getattr(config, 'weight_RMSE', True):
                    loss_delta = criterion3(delta_pred_real, delta_true_real)
                else:
                    loss_delta = criterion1(delta_pred_real, delta_true_real)
                train_loss = loss_delta  # + 0.5 * loss_BG
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                # Standard loss on normalized data
                train_loss = criterion1(output, tgt)
                output_last = output[:, :, :]
                tgt_last = tgt[:, :, :]
                output_numpy = output_last.reshape(-1, 1).detach().cpu().numpy()
                tgt_numpy = tgt_last.reshape(-1, 1).detach().cpu().numpy()
                train_output_inverse = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
                train_target_inverse = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
                train_MSE_loss_inverse = criterion1(train_output_inverse, train_target_inverse)
                train_MAE_loss_inverse = criterion2(train_output_inverse, train_target_inverse)
                train_loss.backward()
                optimizer.step()
                scheduler.step()

            total_train_MSE_loss += train_MSE_loss_inverse.item()
            total_train_MAE_loss += train_MAE_loss_inverse.item()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(validate_dataloader)
            for _, (encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _) in enumerate(pbar):
                encoder_input = encoder_input.to(device)
                encoder_input_mark = encoder_input_mark.to(device)
                decoder_input = decoder_input.to(device)
                decoder_input_mark = decoder_input_mark.to(device)
                tgt = tgt.to(device)

                output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

                validate_dataset = validate_dataloader.dataset
                if hasattr(validate_dataset, "datasets"):
                    inverse_transform_func = validate_dataset.datasets[0].inverse_transform
                    mean = validate_dataset.datasets[0].mean[2]
                    std = validate_dataset.datasets[0].std[2]
                else:
                    inverse_transform_func = validate_dataset.inverse_transform
                    mean = validate_dataset.mean[2]
                    std = validate_dataset.std[2]

                if config.use_delta:
                    # Compute delta-based and absolute BG prediction losses and combine them for training
                    bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                    mean = torch.tensor(mean, device=output.device)
                    std = torch.tensor(std, device=output.device)
                    bg_current_real = bg_current_norm * std + mean
                    delta_pred_real = output * std
                    delta_true_real = tgt * std
                    bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
                    bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
                    validate_MSE_loss_inverse = criterion1(bg_pred_real, bg_true_real)
                    validate_MAE_loss_inverse = criterion2(bg_pred_real, bg_true_real)
                else:
                    output_last = output[:, :, :]
                    tgt_last = tgt[:, :, :]
                    output_numpy = output_last.reshape(-1, 1).detach().cpu().numpy()
                    tgt_numpy = tgt_last.reshape(-1, 1).detach().cpu().numpy()
                    validate_output_inverse = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
                    validate_target_inverse = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
                    validate_MSE_loss_inverse = criterion1(validate_output_inverse, validate_target_inverse)
                    validate_MAE_loss_inverse = criterion2(validate_output_inverse, validate_target_inverse)

                total_validate_MSE_loss += validate_MSE_loss_inverse.item()
                total_validate_MAE_loss += validate_MAE_loss_inverse.item()

        avg_train_RMSE_loss = math.sqrt(total_train_MSE_loss / len(train_dataloader))
        avg_train_MAE_loss = total_train_MAE_loss / len(train_dataloader)
        avg_validate_RMSE_loss = math.sqrt(total_validate_MSE_loss / len(validate_dataloader))
        avg_validate_MAE_loss = total_validate_MAE_loss / len(validate_dataloader)

        if avg_validate_RMSE_loss < validate_min_loss:
            # keep the best weights: on disk for normal runs, in memory for CV
            validate_min_loss = avg_validate_RMSE_loss
            if crossval:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                torch.save(model.state_dict(), model_path)
            early_stop_cnt = 0
            print('Saving best model (epoch = {:4d}, loss = {:.4f})'.format(epoch, validate_min_loss))
        else:
            early_stop_cnt += 1

        if not crossval:
            current_data = {
                'timestamp': current_time,
                'epoch': epoch,
                'train_RMSE': avg_train_RMSE_loss,
                'train_MAE': avg_train_MAE_loss,
                'val_RMSE': avg_validate_RMSE_loss,
                'val_MAE': avg_validate_MAE_loss,
                'learning_rate': last_lr,
                'early_stop_cnt': early_stop_cnt
            }
            df_new = pd.DataFrame([current_data])
            df_new.to_csv(loss_record_csv_path, mode='a', header=not os.path.exists(loss_record_csv_path), index=False)
            record = (
                f"Learning Rate: {last_lr}\n"
                f"Train_RMSE_loss: {avg_train_RMSE_loss:.6f}, Train_MAE_loss: {avg_train_MAE_loss:.6f}\n"
                f"Val_RMSE_loss: {avg_validate_RMSE_loss:.6f}, Val_MAE_loss: {avg_validate_MAE_loss:.6f}\n")
            print(record)
            if epoch % 1 == 0:
                plot_loss_curve(loss_record_csv_path, fig_save_path)

        if early_stop_cnt > config.patience:
            break

    # CV keeps the best weights in memory; normal runs already saved them to disk
    if crossval and best_state is not None:
        model.load_state_dict(best_state)

    print('Finished training after {} epochs, validate_min_loss: {:.6f}'.format(epoch, validate_min_loss))
    return model