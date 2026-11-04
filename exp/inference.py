"""
Inference built straight from run.py — two clean stages.

Stage 1  infer_and_save : run the model, save per-subject predictions/ground truth
                          to npy. No metrics computed here.
    analysis_result/
        {pid}_pred_seed{s}.npy   prediction trajectory [N, pred_len] (mg/dL), one per seed
        {pid}_true.npy           ground-truth trajectory [N, pred_len], seed-independent (saved once)
        all flat in one folder; which pid is test vs zeroshot is decided from the
        patient-id lists returned by load_test_data, not from id length.

Stage 2  compute_metrics_from_npy : read those npy files and compute every metric.
        per-PH point-level : RMSE/MAE take column idx = int(ph/Ts)-1 (the PH-moment forecast);
                             event metrics feed the whole [N, pred_len] window and let
                             event_metrics truncate point-wise by PH.
        whole-window       : RMSE_win/MAE_win, TD/TG (time_gain), window-level events (PH=None).
        results are written to result_subject_{kind}_{pid}.csv and aggregated by summarize.

After changing thresholds/PH/event defs you can recompute stage 2 only with --eval-only,
skipping the model forward entirely.

Usage:
    python inference.py <config.json | experiment_dir> [--root PROJECT_ROOT] [--seed N] [--eval-only]
"""

import os
import glob
import json
import math
import types
import argparse
import numpy as np
import torch
import pandas as pd
from data_provider.T1DMS_Dataset import DataLoader as T1DMS_DataLoader, prepare_T1DMS_data
from data_provider.OhioT1DM_Dataset import prepare_Ohio_data
from data_provider.DiaTrend_Dataset import prepare_DiaTrend_data, prepare_DiaTrend_zeroshot_data
from exp.metrics import time_gain, event_metrics
from utils.tools import set_seed, merge_options

# ----------------------------------------------------------------------
# config + model
# ----------------------------------------------------------------------
def load_config(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "config*.json")))[-1]
    cfg = json.load(open(path))
    # run.py concatenates LOG_DIR + these paths; normalise surrounding whitespace
    for k in ("model_save_path", "loss_record_path", "eval_result_path", "analysis_result_path"):
        if isinstance(cfg.get(k), str):
            cfg[k] = "/" + cfg[k].strip().strip("/")
    return types.SimpleNamespace(**cfg), path


def find_checkpoint(LOG_DIR, config, seed):
    ckpt_dir = os.path.join(LOG_DIR, config.model_save_path.strip("/"))
    exact = os.path.join(ckpt_dir, f"{config.model_name}_seed{seed}.pth")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(ckpt_dir, f"*_seed{seed}.pth")))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"no checkpoint for seed {seed}: {exact}")


def set_model_parameters(config):
    """Build the model named by config.model_name and return (config, model, device)."""
    device = torch.device(config.device)
    seq_len = int(config.seq_len / config.sensor_sampling)
    pred_len = int(config.pred_len / config.sensor_sampling)

    if config.model_name == 'Glucoformer0':
        from models.Glucoformer0.Glucoformer import Glucoformer
        from models.Glucoformer0.Glucoformer_parameters import get_Glucoformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Glucoformer_parser(), args=getattr(config, 'override_args', None))
        seg_len = int(config.seg_len / config.sensor_sampling)
        model = Glucoformer(data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len,
                            output_size=config.c_out, factor=config.factor, d_model=config.d_model,
                            d_ff=config.d_ff, n_heads=config.n_heads, e_layers=config.e_layers,
                            dropout=config.dropout).to(device)

    elif config.model_name == 'Glucoformer':
        from models.Glucoformer.Glucoformer import Glucoformer
        from models.Glucoformer.Glucoformer_parameters import get_Glucoformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Glucoformer_parser(), args=getattr(config, 'override_args', None))
        seg_len = int(config.seg_len / config.sensor_sampling)
        model = Glucoformer(data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len,
                            output_size=config.c_out, factor=config.factor, d_model=config.d_model,
                            d_ff=config.d_ff, n_heads=config.n_heads, e_layers=config.e_layers,
                            dropout=config.dropout,
                            use_sse=config.use_sse, use_etsa=config.use_etsa,
                            use_decoder_self_attn=config.use_decoder_self_attn).to(device)

    elif config.model_name == 'Crossformer':
        from models.Crossformer.cross_former import Crossformer
        from models.Crossformer.Crossformer_parameters import get_Crossformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Crossformer_parser())
        seg_len = int(config.seg_len / config.sensor_sampling)
        model = Crossformer(data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len,
                            output_size=config.c_out, factor=config.factor, d_model=config.d_model,
                            d_ff=config.d_ff, n_heads=config.n_heads, e_layers=config.e_layers,
                            dropout=config.dropout).to(device)

    elif config.model_name == 'PatchTST':
        from models.PatchTST.PatchTST import PatchTST
        from models.PatchTST.PatchTST_parameters import get_PatchTST_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_PatchTST_parser())
        model = PatchTST(config).to(device)

    elif config.model_name == 'TimeXer':
        from models.TimeXer.TimeXer import TimeXer
        from models.TimeXer.TimeXer_parameters import get_TimeXer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_TimeXer_parser())
        model = TimeXer(config).to(device)

    elif config.model_name == 'DLinear':
        from models.DLinear.DLinear_model import DLinear
        from models.DLinear.DLinear_parameters import get_DLinear_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_DLinear_parser())
        model = DLinear(config).to(device)

    elif config.model_name == 'Informer':
        from models.Informer.model import Informer, InformerStack
        from models.Informer.Informer_parameters import get_Informer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Informer_parser())
        if config.model_name == 'Informer':
            model = Informer(enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out,
                             out_len=pred_len, factor=config.factor, d_model=config.d_model,
                             n_heads=config.n_heads, e_layers=config.e_layers,
                             d_layers=config.d_layers, d_ff=config.d_ff, dropout=config.dropout).to(device)
        else:
            model = InformerStack(enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out,
                                 out_len=pred_len, factor=config.factor, d_model=config.d_model,
                                 n_heads=config.n_heads, d_layers=config.d_layers,
                                 d_ff=config.d_ff, dropout=config.dropout).to(device)

    elif config.model_name == 'Transformer':
        from models.Transformer.Transformer_model import Transformer
        from models.Transformer.Transformer_parameters import get_Transformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Transformer_parser())
        model = Transformer(input_size=config.enc_in, pred_length=pred_len, d_model=config.d_model,
                            device=config.device, nhead=config.n_heads,
                            num_encoder_layers=config.e_layers, num_decoder_layers=config.d_layers,
                            dim_feedforward=config.d_ff, output_size=1, dropout=config.dropout).to(device)

    elif config.model_name == 'LSTM':
        from models.LSTM.LSTM_model import LSTM
        from models.LSTM.LSTM_parameters import get_LSTM_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_LSTM_parser())
        model = LSTM(input_size=config.data_dim, hidden_size=config.d_model, num_layers=config.e_layers,
                    forecast_window=pred_len, output_size=config.c_out, dropout=config.dropout).to(device)

    elif config.model_name == 'GRU':
        from models.GRU.GRU_model import GRU
        from models.GRU.GRU_parameters import get_GRU_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_GRU_parser())
        model = GRU(input_size=config.data_dim, hidden_size=config.d_model, num_layers=config.e_layers,
                    forecast_window=pred_len, output_size=config.c_out).to(device)

    elif config.model_name == 'SVR':
        from models.SVR.svr_model import SVRModel
        model = SVRModel(forecast_window=pred_len).to(device)

    elif config.model_name == 'ARIMA':
        from models.ARIMA.arima_model import ARIMAModel
        model = ARIMAModel(forecast_window=pred_len).to(device)

    elif config.model_name == 'ZOH':
        from models.ZOH.zoh_model import ZOHModel
        model = ZOHModel(forecast_window=pred_len).to(device)


    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    return config, model, device


# ----------------------------------------------------------------------
# test data
# ----------------------------------------------------------------------
def load_test_data(config, main_path):
    if "T1DMS" in config.dataset:
        _, _, test_dataset = prepare_T1DMS_data(
            dataset_str=config.dataset, root_dir=main_path,
            seq_length=config.seq_len, label_length=config.label_len,
            pred_length=config.pred_len, use_delta=config.use_delta,
            sensor_sampling=config.sensor_sampling, gaussian=True)
        return T1DMS_DataLoader(test_dataset, config.batch_size, shuffle=False), None, None, None

    if config.dataset == "OhioT1DM":
        _, _, test_dataset, test_patients, scalar = prepare_Ohio_data(
            root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len,
            pred_length=config.pred_len, use_delta=config.use_delta)
        zeroshot_dataset, zeroshot_patients = prepare_DiaTrend_zeroshot_data(
            root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len,
            pred_length=config.pred_len, external_mean=scalar.mean,
            external_std=scalar.std, use_delta=config.use_delta)
        test_dl = T1DMS_DataLoader(test_dataset, config.batch_size, shuffle=False)
        zs_dl = T1DMS_DataLoader(zeroshot_dataset, config.batch_size, shuffle=False)
        return test_dl, test_patients, zs_dl, zeroshot_patients

    if config.dataset == "DiaTrend":
        _, _, test_dataset, test_patients = prepare_DiaTrend_data(
            root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len,
            pred_length=config.pred_len, use_delta=config.use_delta)
        return T1DMS_DataLoader(test_dataset, config.batch_size, shuffle=False), test_patients, None, None

    raise ValueError(f"Unknown dataset: {config.dataset}")


# ----------------------------------------------------------------------
# stage 1: forward one subject -> (pred, true) windows in real glucose space
# ----------------------------------------------------------------------
def forward_subject(dataset, config, ref_dataloader, model, device):
    """Run the model over one subject's windows; return (pred, true) arrays [N, pred_len] (mg/dL)."""
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=ref_dataloader.batch_size, shuffle=False,
        num_workers=ref_dataloader.num_workers, drop_last=False)

    td = ref_dataloader.dataset
    src = td.datasets[0] if hasattr(td, "datasets") else td
    inv_fn, mean, std = src.inverse_transform, src.mean[2], src.std[2]
    std_t = torch.tensor(std, device=device)
    mean_t = torch.tensor(mean, device=device)

    pred_batches, true_batches = [], []
    with torch.no_grad():
        for enc, enc_mark, dec, dec_mark, tgt, _ in dl:
            enc, enc_mark, dec, dec_mark, tgt = (x.to(device) for x in (enc, enc_mark, dec, dec_mark, tgt))
            out = model(enc, enc_mark, dec, dec_mark)

            if config.use_delta:
                # output/tgt are deltas off the current (real) glucose; recover the trajectory
                base = enc[:, -1, -1].unsqueeze(1) * std_t + mean_t          # (B, 1)
                pred = base + (out * std_t).squeeze(-1)                      # (B, pred_len)
                true = base + (tgt * std_t).squeeze(-1)
            else:
                # output/tgt are already absolute glucose in normalised space; just invert
                o = out.reshape(-1, 1).detach().cpu().numpy()
                g = tgt.reshape(-1, 1).detach().cpu().numpy()
                pred = np.asarray(inv_fn(o)).reshape(out.shape[0], -1)       # (B, pred_len)
                true = np.asarray(inv_fn(g)).reshape(tgt.shape[0], -1)

            pred_batches.append(pred.detach().cpu().numpy() if hasattr(pred, 'detach') else pred)
            true_batches.append(true.detach().cpu().numpy() if hasattr(true, 'detach') else true)

    return np.concatenate(pred_batches, axis=0), np.concatenate(true_batches, axis=0)


def infer_and_save(config, LOG_DIR, model, device, test_dataloader, test_patients,
                   zeroshot_dataloader=None, zeroshot_patients=None):
    model.load_state_dict(torch.load(find_checkpoint(LOG_DIR, config, config.seed), map_location=device))
    model.eval()
    out_dir = os.path.join(LOG_DIR, config.analysis_result_path.strip('/'))
    print(f"[stage 1] seed {config.seed}: saving npy -> {out_dir}")

    def subject_pairs(dataloader, patients, concat):
        ds = dataloader.dataset
        if hasattr(ds, "datasets"):
            if concat:   # OhioT1DM test: two sub-datasets (e.g. day/night) per patient
                return [(p, torch.utils.data.ConcatDataset([ds.datasets[2 * i], ds.datasets[2 * i + 1]]))
                        for i, p in enumerate(patients)]
            return [(p, ds.datasets[i]) for i, p in enumerate(patients)]
        return [(p, ds) for p in patients]

    def save_pairs(pairs, d):
        os.makedirs(d, exist_ok=True)
        for pid, sub_ds in pairs:
            pred, true = forward_subject(sub_ds, config, test_dataloader, model, device)
            np.save(f"{d}/{pid}_pred_seed{config.seed}.npy", pred)
            if not os.path.exists(f"{d}/{pid}_true.npy"):     # ground truth is seed-independent
                np.save(f"{d}/{pid}_true.npy", true)

    save_pairs(subject_pairs(test_dataloader, test_patients, config.dataset == 'OhioT1DM'), out_dir)
    if zeroshot_dataloader is not None and zeroshot_patients:
        save_pairs(subject_pairs(zeroshot_dataloader, zeroshot_patients, False), out_dir)


# ----------------------------------------------------------------------
# stage 2: all metrics for one (pred, true) pair -> a flat dict
# ----------------------------------------------------------------------
def metric_row(pred, true, config, PH):
    Ts = config.sensor_sampling
    low, high = 70.0, 180.0
    row = {}

    for ph in PH:
        idx = int(ph / Ts) - 1
        p, t = pred[:, idx], true[:, idx]
        row[f'RMSE_{ph}'] = math.sqrt(float(np.mean((p - t) ** 2)))
        row[f'MAE_{ph}'] = float(np.mean(np.abs(p - t)))
        for ev_type, tag in (('hypo', 'Hypo'), ('hyper', 'Hyper')):
            m = event_metrics(true, pred, event_type=ev_type, low=low, high=high, Ts=Ts, PH=ph)
            row.update({f'{tag}_{k}_{ph}': m[k] for k in ('Sens', 'Spec', 'Prec', 'F1', 'Acc')})

    row['RMSE_win'] = math.sqrt(float(np.mean((pred - true) ** 2)))
    row['MAE_win'] = float(np.mean(np.abs(pred - true)))
    for ph in PH:
        TD, TG = time_gain(true, pred, PH=ph, Ts=Ts)
        row[f'TD_{ph}'], row[f'TG_{ph}'] = TD, TG
    for ev_type, tag in (('hypo', 'Hypo'), ('hyper', 'Hyper')):
        m = event_metrics(true, pred, event_type=ev_type, low=low, high=high,
                          Ts=Ts, min_dur_min=15.0, PH=None)
        row.update({f'{tag}_{k}_win': m[k] for k in ('Sens', 'Spec', 'Prec', 'F1', 'Acc')})

    return row


def compute_metrics_from_npy(config, LOG_DIR, PH=None, test_patients=None, zeroshot_patients=None):
    PH = list(PH or config.PH)
    base = os.path.join(LOG_DIR, config.analysis_result_path.strip('/'))
    # rigorous split: which pid is test vs zeroshot comes straight from the
    # patient-id lists returned by load_test_data -- never from id length.
    test_set = {str(p) for p in (test_patients or [])}
    zs_set = {str(p) for p in (zeroshot_patients or [])}

    rows = {}
    for pred_path in sorted(glob.glob(f"{base}/*_pred_seed*.npy")):
        name = os.path.basename(pred_path)[:-4]              # {pid}_pred_seed{s}
        pid, seed = name.rsplit("_pred_seed", 1)
        kind = 'zeroshot' if pid in zs_set else 'test'
        pred, true = np.load(pred_path), np.load(f"{base}/{pid}_true.npy")
        row = {'run_seed': int(seed), 'patient_id': pid}
        row.update(metric_row(pred, true, config, PH))
        rows.setdefault((kind, pid), []).append(row)
    return rows


# ----------------------------------------------------------------------
# result columns (single source of truth; 'run_seed' for subject csv, 'patient_id' for mean csv)
# ----------------------------------------------------------------------
def result_columns(PH, first='run_seed'):
    ev_bases = ['Hypo_Sens', 'Hypo_Spec', 'Hypo_Prec', 'Hypo_F1', 'Hypo_Acc',
                'Hyper_Sens', 'Hyper_Spec', 'Hyper_Prec', 'Hyper_F1', 'Hyper_Acc']
    return ([first]
            + [f'{m}_{ph}' for ph in PH for m in ('RMSE', 'MAE')]
            + ['RMSE_win', 'MAE_win']
            + [f'TD_{ph}' for ph in PH] + [f'TG_{ph}' for ph in PH]
            + [f'{ev}_{ph}' for ph in PH for ev in ev_bases]
            + [f'{ev}_win' for ev in ev_bases])


# ----------------------------------------------------------------------
# write per-subject seed csvs and aggregate them into the per-kind mean csv
# ----------------------------------------------------------------------
def summarize(config, LOG_DIR, subject_rows):
    eval_dir = os.path.join(LOG_DIR, config.eval_result_path.strip('/'))
    os.makedirs(eval_dir, exist_ok=True)
    PH = list(config.PH)
    subject_cols = result_columns(PH, 'run_seed')
    mean_cols = result_columns(PH, 'patient_id')
    ev_bases = ['Hypo_Sens', 'Hypo_Spec', 'Hypo_Prec', 'Hypo_F1', 'Hypo_Acc',
                'Hyper_Sens', 'Hyper_Spec', 'Hyper_Prec', 'Hyper_F1', 'Hyper_Acc']

    def stat(series):
        series = series.dropna()
        if not len(series):
            return 'NaN'
        spread = f"±{series.std():.4f}" if len(series) > 1 else "±0.0000"
        return f"{series.mean():.4f}{spread}"

    specs = [('test', 'result_mean_test.csv')]
    if config.dataset == 'OhioT1DM':
        specs.append(('zeroshot', 'result_mean_zeroshot.csv'))

    for kind, mean_file in specs:
        own = {pid: r for (kk, pid), r in subject_rows.items() if kk == kind}
        if not own:
            print(f"\nNo {kind} results to summarize.")
            continue

        patient_rows = []
        for pid, rows in sorted(own.items()):
            df = pd.DataFrame(rows)[subject_cols]
            df.to_csv(f"{eval_dir}/result_subject_{kind}_{pid}.csv", index=False)

            r = {'patient_id': pid}
            for ph in PH:
                for m in ('RMSE', 'MAE', 'TD', 'TG'):
                    r[f'{m}_{ph}'] = stat(df[f'{m}_{ph}'])
            for ev in ev_bases:
                for ph in PH:
                    r[f'{ev}_{ph}'] = stat(df.get(f'{ev}_{ph}'))
                r[f'{ev}_win'] = stat(df.get(f'{ev}_win'))
            r['RMSE_win'] = stat(df['RMSE_win'])
            r['MAE_win'] = stat(df['MAE_win'])
            patient_rows.append(r)

        # overall row: average the per-patient means
        overall = {'patient_id': 'All_Patients'}
        for c in mean_cols:
            if c == 'patient_id':
                continue
            vals = []
            for r in patient_rows:
                tok = r[c].split('±')[0]
                if tok != 'NaN':
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
            overall[c] = stat(pd.Series(vals)) if vals else 'NaN'

        summary = pd.DataFrame(patient_rows + [overall])[mean_cols]
        summary.to_csv(f"{eval_dir}/{mean_file}", index=False)
        print(f"\n{kind} average by patient and overall:")
        print(summary.to_string(index=False))


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def run_inference(config_json, main_path=None, seed=None, eval_only=False):
    main_path = main_path or os.getcwd()
    config, cfg_path = load_config(config_json)
    LOG_DIR = os.path.dirname(cfg_path)
    seeds = [seed] if seed is not None else range(config.repeat_times)
    print(f"config : {cfg_path}\nmodel  : {config.model_name} | dataset: {config.dataset} | seeds: {list(seeds)}")

    test_dl, test_patients, zs_dl, zs_patients = load_test_data(config, main_path)

    if not eval_only:
        analysis_dir = os.path.join(LOG_DIR, config.analysis_result_path.strip('/'))
        if os.path.isdir(analysis_dir):
            import shutil
            shutil.rmtree(analysis_dir)        # drop stale npy before a fresh forward
        for s in seeds:
            config.seed = s
            set_seed(s)
            config.experiment_dir = os.path.relpath(LOG_DIR, main_path)
            _, model, device = set_model_parameters(config)
            infer_and_save(config, LOG_DIR, model, device, test_dl, test_patients,
                           zeroshot_dataloader=zs_dl, zeroshot_patients=zs_patients)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # metrics split test/zeroshot using the patient-id lists from load_test_data
    subject_rows = compute_metrics_from_npy(config, LOG_DIR, config.PH, test_patients, zs_patients)
    if not subject_rows:
        print("no npy results found; nothing to evaluate")
        return
    summarize(config, LOG_DIR, subject_rows)
    print(f"\ndone. results in {os.path.join(LOG_DIR, config.analysis_result_path.strip('/'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_json", help="path to config*.json or an experiment dir")
    ap.add_argument("--root", default=None, help="project root (data location), default cwd")
    ap.add_argument("--seed", type=int, default=None, help="run a single seed only")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip inference; recompute metrics from existing npy files")
    args = ap.parse_args()
    run_inference(args.config_json, main_path=args.root, seed=args.seed, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
