import torch
import torch.nn as nn
import numpy as np
import math

class MultiHorizonMSELoss(nn.Module):
    def __init__(self, pred_len, mode='power', alpha=0.1, beta=0.7):
        super().__init__()
        self.pred_len = pred_len

        h = torch.arange(1, pred_len + 1).float()

        if mode == 'uniform':
            weights = torch.ones_like(h)

        elif mode == 'decay':
            weights = torch.exp(-alpha * h)

        elif mode == 'linear':
            weights = h

        elif mode == 'power': 
            weights = h ** beta

        else:
            raise ValueError("Unknown mode")

        weights = weights / weights.sum()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        loss = (pred - target) ** 2
        weights = self.weights.to(loss.device) 
        loss = loss * weights 

        return loss.mean()


def time_gain(y_true, y_pred, PH, Ts, method="correlation"):
    """Compute TD and TG via full-sequence cross-correlation.
    y_true,y_pred: 1D series or 2D (N,L) sliding window matrix.
    PH: prediction horizon (min), Ts: sampling interval (min).
    method: "correlation"(recommended) or "mse".
    Return (TD, TG)."""
    yt, yp = map(np.asarray, (y_true, y_pred))
    if yt.ndim == 2:
        i = min(max(int(round(PH / Ts)) - 1, 0), yt.shape[1] - 1)
        yt, yp = yt[:, i], yp[:, i]
    if yt.ndim != 1 or yt.shape != yp.shape:
        raise ValueError("Input must be equal-length 1D or 2D (N,L) array")
    kmax = max(1, min(int(round(PH / Ts)), len(yt) - 2))
    TD = 0.0
    if kmax >= 1:
        ks = np.arange(-kmax, kmax + 1)
        s = [_score(yt, yp, int(k), method) for k in ks]
        TD = int(ks[int(np.nanargmax(s))]) * Ts
    return TD, PH - TD

def _score(yt, yp, k, method):
    """Compute matching score for shift k. k>0: prediction lags ground truth."""
    t, p = (yt[:-k], yp[k:]) if k >= 0 else (yt[-k:], yp[:k])
    if len(t) < 3: return np.nan
    if method == "correlation":
        a, b = t - t.mean(), p - p.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return (a * b).sum() / d if d > 1e-12 else np.nan
    return -float(np.mean((t - p) ** 2))


def event_metrics(true_arr, pred_arr, event_type='hypo', low=70.0, high=180.0,
                  alarm_low=90, Ts=5.0, min_dur_min=15.0, PH=None):
    """Unified point-level(PH≠None) / window-level(PH=None) event classification.
    hypo: true BG<low, alarm pred<alarm_low; hyper: true BG>high, alarm pred>high.
    Window-level: true event requires continuous violation >=min_dur_min; alarm triggers if any point violates."""
    alarm_low = low if alarm_low is None else alarm_low
    y_true, y_pred = np.asarray(true_arr, float), np.asarray(pred_arr, float)
    if PH is not None:
        n = int(round(PH / Ts)); y_true, y_pred = y_true[..., :n], y_pred[..., :n]
    if event_type == 'hypo': real, alarm = y_true < low, y_pred < alarm_low
    elif event_type == 'hyper': real, alarm = y_true > high, y_pred > high
    else: raise ValueError("event_type must be 'hypo' or 'hyper'")
    if PH is None: # window-level metric
        if real.ndim != 2: raise ValueError("Window mode needs 2D (N,L) input")
        cs = np.cumsum(real.astype(int), axis=1)
        runs = cs - np.cumsum(np.where(real, 0, cs), axis=1)
        yt = (runs.max(axis=1) >= max(1, int(round(min_dur_min / Ts)))).astype(int)
        yp = alarm.any(axis=1).astype(int)
    else: # point-level metric
        yt, yp = real.ravel().astype(int), alarm.ravel().astype(int)
    tp,tn,fp,fn = int(np.sum((yt==1)&(yp==1))),int(np.sum((yt==0)&(yp==0))),int(np.sum((yt==0)&(yp==1))),int(np.sum((yt==1)&(yp==0)))
    N = yt.size
    sens = tp/(tp+fn) if (tp+fn) else math.nan
    spec = tn/(tn+fp) if (tn+fp) else math.nan
    prec = tp/(tp+fp) if (tp+fp) else math.nan
    f1 = 2*prec*sens/(prec+sens) if not (math.isnan(prec) or math.isnan(sens)) and (prec+sens) else math.nan
    return {'Sens':sens,'Spec':spec,'Prec':prec,'F1':f1,'Acc':(tp+tn)/N if N else math.nan}

