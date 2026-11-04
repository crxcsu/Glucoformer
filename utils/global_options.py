import argparse
import torch
from typing import Optional


class global_options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='global')
        self.opt = None

    def _initial(self):
        # ========== Basic Task Configuration ==========
        self.parser.add_argument('--model_name', type=str, default='Glucoformer', help='model name')
        self.parser.add_argument('--dataset', type=str, default='T1DMS', help='dataset name, eg., T1DMS, OhioT1DM, DiaTrend, Shanghai')
        self.parser.add_argument('--experiment_dir', type=str, default=None, help='directory to save experiment results, including model checkpoints and logs')
        self.parser.add_argument('--repeat_times', type=int, default=10, help='number of times to repeat the experiment')
        self.parser.add_argument('--PH', type=list, default=[30, 60, 90, 120], help='prediction horizons (minutes)')

        # ========== Dataset and DataLoader Configuration ==========
        self.parser.add_argument('--seq_len', type=int, default=480, help='input sequence length')
        self.parser.add_argument('--label_len', type=int, default=0, help='start token length')
        self.parser.add_argument('--pred_len', type=int, default=120, help='prediction sequence length')
        self.parser.add_argument('--stride', type=int, default=1, help='the step of stride window')
        self.parser.add_argument('--sensor_sampling', type=int, default=5, help='sensor sampling interval (minutes)')

        
        # ========== Optimizer Configuration ==========
        self.parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
        self.parser.add_argument('--use_delta', type=bool, default=True, help='whether to use delta-based prediction target')
        self.parser.add_argument('--weight_RMSE', type=bool, default=True, help='whether to weight RMSE in loss function')
        # self.parser.add_argument('--step_size', type=int, default=1, help='Period of learning rate decay')
        # self.parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay for pretraining')

        # ========== Model saving and result path configuration ========== 
        self.parser.add_argument('--model_save_path', type=str, default='/checkpoints', help='path to save trained model')
        self.parser.add_argument('--loss_record_path', type=str, default='/loss_record', help='path to loss records')
        self.parser.add_argument('--eval_result_path', type=str, default='/eval_result', help='path to evaluation results')
        self.parser.add_argument('--analysis_result_path', type=str, default='/analysis_result', help='path to analysis results')

        # ========== Device Configuration ==========
        self.parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation: 'cpu' or 'cuda'")

    def parse(self, args=None):
        """Parse command line arguments"""
        self._initial()
        if args is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(args)
        return self.opt