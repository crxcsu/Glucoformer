import os
import json
import datetime
import multiprocessing

import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader as TorchDataLoader

from data_provider.T1DMS_Dataset import DataLoader
from data_provider.OhioT1DM_Dataset import prepare_Ohio_LOSO
from utils.tools import set_seed, setup_folders

# inference.py is the single source of truth for model building, the forward
# pass that produces real-glucose npy, and all metric computation / aggregation.
from exp.inference import (set_model_parameters, forward_subject,
                       compute_metrics_from_npy, summarize)
# training lives in exp.train_val; CV keeps the best weights in memory (no disk writes)
from exp.train_val import train


# =============================================================================
# top-level worker (must be module-level to be pickled into the child process)
# trains one fold and writes that fold's held-out subject npy; returns its pid.
# =============================================================================
def worker_fold(fold_index, config, LOG_DIR, main_path):
    set_seed(config.seed)
    config, model, device = set_model_parameters(config)

    folds = prepare_Ohio_LOSO(
        root_dir=main_path,
        seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len,
        use_delta=config.use_delta, gaussian=False)
    fold = folds[fold_index]

    train_loader = DataLoader(fold["train_set"], config.batch_size, shuffle=True)
    val_loader = DataLoader(fold["val_set"], config.batch_size, shuffle=False)
    test_loader = TorchDataLoader(fold["test_set"], batch_size=config.batch_size,
                                  shuffle=False, drop_last=False)

    # train (best weights kept in memory; nothing saved to disk)
    model = train(config, LOG_DIR, train_loader, val_loader, device, model, crossval=True)

    analysis_dir = os.path.join(LOG_DIR, config.analysis_result_path.strip('/'))
    os.makedirs(analysis_dir, exist_ok=True)

    # ---- Stage 1 (inference.py): forward + save npy, no metrics here ----
    pid = fold["test_patient"]
    pred, true = forward_subject(fold["test_set"], config, test_loader, model, device)
    np.save(f"{analysis_dir}/{pid}_pred_seed{config.seed}.npy", pred)
    np.save(f"{analysis_dir}/{pid}_true.npy", true)
    return pid


# =============================================================================
# main entry
# =============================================================================
def main(config, main_path=os.getcwd()):
    # CV trains each fold once per seed. Repeat is forced to 1 for now; raising
    # config.repeat_times later simply re-runs the whole LOSO CV for each seed
    # (each seed writes its own *_pred_seed{s}.npy), no other change needed.

    # prediction horizons (fall back to a default if absent)
    config.PH = getattr(config, "PH", [30, 60, 90, 120])

    # merge model hyper-parameters once while experiment_dir is still None
    config, _, _ = set_model_parameters(config)

    if config.experiment_dir is None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_id = 'cv_{}_{}_pretrain{}_bs{}_lr{:.1e}_dm{}_el{}_{}'.format(
            config.dataset,
            config.model_name,
            1 if (config.pretrain_model_pth is not None and config.pretrain_model_pth != 'None') else 0,
            config.batch_size,
            config.lr,
            getattr(config, 'd_model', 0),
            getattr(config, 'e_layers', 0),
            time_stamp)
        config.experiment_dir = os.path.join("results", folder_id)

    LOG_DIR = os.path.join(main_path, config.experiment_dir)
    setup_folders(LOG_DIR, config)  # clear=False, directory layout unchanged

    with open(os.path.join(LOG_DIR, "config.json"), 'w') as f:
        json.dump(vars(config), f, indent=4)

    print("\n" + "=" * 30 + " Configuration " + "=" * 30)
    print(json.dumps(vars(config), indent=4))
    print("=" * 75 + "\n")

    config.repeat_times = 1
    for seed in range(0, config.repeat_times):
        config.seed = seed
        set_seed(config.seed)
        if config.dataset == "OhioT1DM":
            run_ohio_loso(config, LOG_DIR, main_path)
        else:
            raise ValueError(f"current implementation only supports OhioT1DM, got dataset={config.dataset}")


# =============================================================================
# OhioT1DM: 12-fold LOSO, 4 folds in flight at a time
# =============================================================================
def run_ohio_loso(config, LOG_DIR, main_path):
    n_folds = 12
    max_workers = 4  # run 4 folds concurrently

    # spawn avoids CUDA + fork issues and works on CPU too
    ctx = multiprocessing.get_context("spawn")

    held_out = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(worker_fold, i, config, LOG_DIR, main_path)
                   for i in range(n_folds)]
        for fut in tqdm.tqdm(futures, total=n_folds, desc="LOSO folds"):
            held_out.append(fut.result())  # pid of the subject this fold held out

    # ---- Stage 2 (inference.py): read every npy once, compute + aggregate ----
    subject_rows = compute_metrics_from_npy(
        config, LOG_DIR, PH=config.PH, test_patients=held_out, zeroshot_patients=None)
    summarize(config, LOG_DIR, subject_rows)
