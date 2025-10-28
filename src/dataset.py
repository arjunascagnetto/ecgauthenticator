import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    """Dataset per autoencoder - ritorna singoli esami come tensori"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file: path al file CSV (train/val/test)
        """
        self.df = pd.read_csv(csv_file)
        self.ecg_features = ['VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval',
                             'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset',
                             'POnset', 'POffset', 'TOffset']

        self.data = self.df[self.ecg_features].values.astype(np.float32)
        self.patient_ids = self.df['PatientID'].values
        self.exam_ids = self.df['ExamID'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Ritorna: (esame as tensor, patient_id, exam_id)"""
        exam = torch.from_numpy(self.data[idx])
        patient_id = self.patient_ids[idx]
        exam_id = self.exam_ids[idx]

        return exam, patient_id, exam_id

    def get_feature_names(self):
        """Ritorna nomi delle features"""
        return self.ecg_features

    def get_num_features(self):
        """Ritorna numero di features"""
        return len(self.ecg_features)
