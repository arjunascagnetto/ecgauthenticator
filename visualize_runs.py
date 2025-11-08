"""
Visualizer per training runs - PyQt5 GUI
Legge history.csv da runs_v2 e plotta metriche di training
"""

import sys
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QTabWidget, QLabel, QStatusBar, QCheckBox,
    QTextEdit, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


class TrainingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Training Run Visualizer")
        self.setGeometry(100, 100, 1400, 900)

        self.runs_dir = Path("runs_v2")
        self.current_run = None
        self.current_run_path = None
        self.history_df = None
        self.results_json = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_evaluation = False

        self.init_ui()
        self.load_runs()

    def init_ui(self):
        """Inizializza UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principale
        main_layout = QVBoxLayout(central_widget)

        # Top bar: selezione run e pulsante load
        top_layout = QHBoxLayout()

        label = QLabel("Seleziona Run:")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        top_layout.addWidget(label)

        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(400)
        top_layout.addWidget(self.run_combo)

        self.load_btn = QPushButton("Load Run")
        self.load_btn.setMinimumWidth(100)
        self.load_btn.clicked.connect(self.load_selected_run)
        top_layout.addWidget(self.load_btn)

        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Training Metrics
        self.training_tab = QWidget()
        self.training_layout = QVBoxLayout(self.training_tab)
        self.tabs.addTab(self.training_tab, "Training Metrics")

        # Figure per matplotlib
        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.training_layout.addWidget(self.canvas)

        # Tab 2: Evaluation
        self.eval_tab = QWidget()
        self.eval_layout = QVBoxLayout(self.eval_tab)
        self.tabs.addTab(self.eval_tab, "Evaluation")

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Epoch:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setMinimum(1)
        self.epoch_spinbox.setMaximum(50)
        self.epoch_spinbox.setValue(1)
        controls_layout.addWidget(self.epoch_spinbox)

        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Datasets:"))

        self.train_cb = QCheckBox("Train")
        self.train_cb.setChecked(True)
        controls_layout.addWidget(self.train_cb)

        self.val_cb = QCheckBox("Val")
        self.val_cb.setChecked(True)
        controls_layout.addWidget(self.val_cb)

        self.test_cb = QCheckBox("Test")
        self.test_cb.setChecked(True)
        controls_layout.addWidget(self.test_cb)

        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Sampling:"))
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems([
            "All samples",
            "1000 samples",
            "2000 samples",
            "5000 samples",
            "10000 samples",
            "Random patients (all exams)"
        ])
        controls_layout.addWidget(self.sampling_combo)

        controls_layout.addStretch()

        self.eval_btn = QPushButton("Run Evaluation")
        self.eval_btn.setMinimumWidth(120)
        self.eval_btn.clicked.connect(self.run_evaluation)
        controls_layout.addWidget(self.eval_btn)

        self.stop_eval_btn = QPushButton("Stop Evaluation")
        self.stop_eval_btn.setMinimumWidth(120)
        self.stop_eval_btn.setEnabled(False)
        self.stop_eval_btn.clicked.connect(self.stop_evaluation_clicked)
        controls_layout.addWidget(self.stop_eval_btn)

        self.eval_layout.addLayout(controls_layout)

        # Results text
        self.eval_results = QTextEdit()
        self.eval_results.setReadOnly(True)
        self.eval_results.setFont(QFont("Courier", 9))
        self.eval_layout.addWidget(self.eval_results)

        # Tab 3: Best Epoch Stats
        self.best_epoch_tab = QWidget()
        self.best_epoch_layout = QVBoxLayout(self.best_epoch_tab)
        self.tabs.addTab(self.best_epoch_tab, "Best Epoch Stats")

        # Stats info
        self.best_epoch_info = QLabel("Load a run to see best epoch stats")
        self.best_epoch_info.setFont(QFont("Arial", 11, QFont.Bold))
        self.best_epoch_layout.addWidget(self.best_epoch_info)

        self.best_epoch_layout.addSpacing(10)

        # Controls
        best_controls_layout = QHBoxLayout()

        best_controls_layout.addWidget(QLabel("Datasets:"))

        self.best_train_cb = QCheckBox("Train")
        self.best_train_cb.setChecked(True)
        best_controls_layout.addWidget(self.best_train_cb)

        self.best_val_cb = QCheckBox("Val")
        self.best_val_cb.setChecked(True)
        best_controls_layout.addWidget(self.best_val_cb)

        self.best_test_cb = QCheckBox("Test")
        self.best_test_cb.setChecked(True)
        best_controls_layout.addWidget(self.best_test_cb)

        best_controls_layout.addSpacing(20)
        best_controls_layout.addWidget(QLabel("Sampling:"))
        self.best_sampling_combo = QComboBox()
        self.best_sampling_combo.addItems([
            "All samples",
            "1000 samples",
            "2000 samples",
            "5000 samples",
            "10000 samples",
            "Random patients (all exams)"
        ])
        best_controls_layout.addWidget(self.best_sampling_combo)

        best_controls_layout.addStretch()

        self.best_eval_btn = QPushButton("Run Evaluation")
        self.best_eval_btn.setMinimumWidth(120)
        self.best_eval_btn.clicked.connect(self.run_best_epoch_evaluation)
        best_controls_layout.addWidget(self.best_eval_btn)

        self.best_stop_eval_btn = QPushButton("Stop Evaluation")
        self.best_stop_eval_btn.setMinimumWidth(120)
        self.best_stop_eval_btn.setEnabled(False)
        self.best_stop_eval_btn.clicked.connect(self.stop_evaluation_clicked)
        best_controls_layout.addWidget(self.best_stop_eval_btn)

        self.best_epoch_layout.addLayout(best_controls_layout)

        # Results text
        self.best_epoch_results = QTextEdit()
        self.best_epoch_results.setReadOnly(True)
        self.best_epoch_results.setFont(QFont("Courier", 9))
        self.best_epoch_layout.addWidget(self.best_epoch_results)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_runs(self):
        """Carica lista di run da runs_v2/"""
        if not self.runs_dir.exists():
            self.statusBar().showMessage("❌ runs_v2/ directory not found")
            return

        runs = sorted([d for d in self.runs_dir.iterdir() if d.is_dir()])

        if not runs:
            self.statusBar().showMessage("❌ No runs found in runs_v2/")
            return

        self.run_combo.clear()
        for run in runs:
            self.run_combo.addItem(run.name, userData=str(run))

        self.statusBar().showMessage(f"Found {len(runs)} runs")

    def load_selected_run(self):
        """Carica la run selezionata e plotta metriche"""
        run_name = self.run_combo.currentText()
        run_path = Path(self.run_combo.currentData())

        history_file = run_path / "history.csv"
        config_file = Path(run_path).parent.parent / "train_config_v2.yaml"

        if not history_file.exists():
            self.statusBar().showMessage(f"❌ history.csv not found in {run_name}")
            return

        try:
            # Carica history
            self.history_df = pd.read_csv(history_file)
            self.current_run = run_name
            self.current_run_path = run_path

            # Carica results
            results_file = run_path / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.results_json = json.load(f)
                    best_epoch = self.results_json.get('best_epoch', 'N/A')
                    best_ch = self.results_json.get('best_ch', 'N/A')
                    final_db = self.results_json.get('final_db', 'N/A')

                    info_text = f"Best Epoch: {best_epoch} | Best CH: {best_ch:.4f} | Final DB: {final_db:.4f}"
                    self.best_epoch_info.setText(info_text)

            # Plotta
            self.plot_training_metrics()

            self.statusBar().showMessage(f"✓ Loaded: {run_name} ({len(self.history_df)} epochs)")

        except Exception as e:
            self.statusBar().showMessage(f"❌ Error loading run: {str(e)}")

    def add_trend_line(self, ax, x, y, degree=2, color='black', alpha=0.5, linestyle='--', label='Trend'):
        """Aggiunge linea di regressione polinomiale ai plot"""
        if len(x) < 2:
            return

        # Fit polinomiale
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = p(x_smooth)

        ax.plot(x_smooth, y_smooth, color=color, linestyle=linestyle, alpha=alpha, linewidth=2, label=label)

    def plot_training_metrics(self):
        """Plotta le metriche di training"""
        if self.history_df is None:
            return

        self.figure.clear()

        # 2x3 layout
        axes = self.figure.subplots(2, 3)
        axes = axes.flatten()

        epochs = self.history_df['epoch'].values

        # Plot 1: Training Loss
        ax = axes[0]
        ax.plot(epochs, self.history_df['train_loss'], 'b-', linewidth=2, label='Loss')
        self.add_trend_line(ax, epochs, self.history_df['train_loss'].values, degree=2, color='darkblue', label='Trend')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Intra distance
        ax = axes[1]
        ax.plot(epochs, self.history_df['train_intra_dist'], 'g-', linewidth=2, label='Intra')
        self.add_trend_line(ax, epochs, self.history_df['train_intra_dist'].values, degree=2, color='darkgreen', label='Trend')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.set_title('Intra-Patient Distance (Same patient)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Inter distance
        ax = axes[2]
        ax.plot(epochs, self.history_df['train_inter_dist'], 'r-', linewidth=2, label='Inter')
        self.add_trend_line(ax, epochs, self.history_df['train_inter_dist'].values, degree=2, color='darkred', label='Trend')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.set_title('Inter-Patient Distance (Different patient)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 4: Distance Ratio
        ax = axes[3]
        ax.plot(epochs, self.history_df['train_dist_ratio'], 'm-', linewidth=2, label='Ratio')
        self.add_trend_line(ax, epochs, self.history_df['train_dist_ratio'].values, degree=2, color='purple', label='Trend')
        ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.3, label='Baseline (1.0x)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio (Inter/Intra)')
        ax.set_title('Distance Ratio (higher is better)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 5: Validation CH Score
        ax = axes[4]
        ax.plot(epochs, self.history_df['val_ch'], 'c-', linewidth=2, marker='o', markersize=4, label='CH')
        self.add_trend_line(ax, epochs, self.history_df['val_ch'].values, degree=2, color='darkcyan', label='Trend')
        best_epoch = self.history_df.loc[self.history_df['val_ch'].idxmax(), 'epoch']
        best_ch = self.history_df['val_ch'].max()
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_epoch:.0f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CH Score')
        ax.set_title('Validation CH Score (higher is better)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 6: Validation DB Index
        ax = axes[5]
        ax.plot(epochs, self.history_df['val_db'], 'orange', linewidth=2, marker='s', markersize=4, label='DB')
        self.add_trend_line(ax, epochs, self.history_df['val_db'].values, degree=2, color='darkorange', label='Trend')
        best_db_epoch = self.history_df.loc[self.history_df['val_db'].idxmin(), 'epoch']
        best_db = self.history_df['val_db'].min()
        ax.axvline(x=best_db_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_db_epoch:.0f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('DB Index')
        ax.set_title('Validation DB Index (lower is better)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Title generale
        self.figure.suptitle(f'Training Run: {self.current_run}', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

        self.canvas.draw()

    def stop_evaluation_clicked(self):
        """Arresta la valutazione in corso"""
        self.stop_evaluation = True
        self.statusBar().showMessage("Stopping evaluation...")

    def run_evaluation(self):
        """Esegue valutazione del modello su dataset selezionati"""
        if self.current_run_path is None:
            self.statusBar().showMessage("❌ Nessuna run caricata")
            return

        epoch = self.epoch_spinbox.value()
        model_path = self.current_run_path / "models" / f"model_epoch_{epoch:04d}.pth"

        if not model_path.exists():
            self.statusBar().showMessage(f"❌ Model epoch {epoch} not found")
            return

        datasets = []
        if self.train_cb.isChecked():
            datasets.append(('train', 'data/ECG/train.csv'))
        if self.val_cb.isChecked():
            datasets.append(('val', 'data/ECG/val.csv'))
        if self.test_cb.isChecked():
            datasets.append(('test', 'data/ECG/test.csv'))

        if not datasets:
            self.statusBar().showMessage("❌ Seleziona almeno un dataset")
            return

        self.stop_evaluation = False
        self.eval_btn.setEnabled(False)
        self.stop_eval_btn.setEnabled(True)

        self.statusBar().showMessage(f"Computing evaluation for epoch {epoch}...")
        self.eval_results.setText("Computing...\n")

        try:
            # Carica encoder dal config
            from src.ecg_encoder import ECGEncoder

            config_file = self.current_run_path.parent.parent / "train_config_v2.yaml"
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Carica modello
            encoder = ECGEncoder(
                input_dim=config['encoder']['input_dim'],
                hidden_dims=config['encoder'].get('hidden_dims', [20]),
                embedding_dim=config['encoder']['embedding_dim'],
                dropout=config['encoder']['dropout'],
                normalize=config['encoder']['normalize']
            ).to(self.device)

            encoder.load_state_dict(torch.load(model_path, map_location=self.device))
            encoder.eval()

            sampling_str = self.sampling_combo.currentText()
            results_text = f"=== Evaluation Results - Epoch {epoch} ===\n"
            results_text += f"Sampling: {sampling_str}\n\n"

            for dataset_name, csv_path in datasets:
                if self.stop_evaluation:
                    break

                csv_full_path = Path(csv_path) if Path(csv_path).is_absolute() else self.current_run_path.parent.parent / csv_path

                if not csv_full_path.exists():
                    results_text += f"\n❌ {dataset_name}: File not found ({csv_full_path})\n"
                    continue

                # Carica dataset
                df = pd.read_csv(csv_full_path)
                feature_cols = [
                    'VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                    'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset'
                ]
                features = df[feature_cols].values.astype(np.float32)
                patient_ids = df['PatientID'].values

                # Normalizza features (usando scaler da training)
                scaler_mean = features.mean(axis=0)
                scaler_std = features.std(axis=0)
                features = (features - scaler_mean) / (scaler_std + 1e-8)

                # Calcola embeddings
                embeddings = self._get_embeddings(encoder, features)

                # Campionamento
                sample_indices = self._get_sample_indices(embeddings, patient_ids, sampling_str)

                # Calcola metriche
                roc_auc, top1, top5, top10 = self._compute_metrics(embeddings, patient_ids, sample_indices)

                results_text += f"\n{'='*40}\n"
                results_text += f"{dataset_name.upper()}\n"
                results_text += f"{'='*40}\n"
                results_text += f"ROC-AUC:       {roc_auc:.4f}\n"
                results_text += f"Ranking@1:     {top1:.2f}%\n"
                results_text += f"Ranking@5:     {top5:.2f}%\n"
                results_text += f"Ranking@10:    {top10:.2f}%\n"

            self.eval_results.setText(results_text)
            if self.stop_evaluation:
                self.statusBar().showMessage("Evaluation stopped by user")
            else:
                self.statusBar().showMessage(f"✓ Evaluation complete")

        except Exception as e:
            self.eval_results.setText(f"❌ Error: {str(e)}")
            self.statusBar().showMessage(f"❌ Error: {str(e)}")

        finally:
            self.eval_btn.setEnabled(True)
            self.stop_eval_btn.setEnabled(False)

    @torch.no_grad()
    def _get_embeddings(self, encoder, features, batch_size=256):
        """Calcola embeddings per tutte le features"""
        all_embeddings = []

        for batch_start in range(0, len(features), batch_size):
            batch_end = min(batch_start + batch_size, len(features))
            batch_features = features[batch_start:batch_end]

            x = torch.from_numpy(batch_features).float().to(self.device)
            embeddings = encoder(x).cpu().numpy()
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def _get_sample_indices(self, embeddings, patient_ids, sampling_str):
        """Ritorna indici campionati in base alla scelta"""
        total_samples = len(embeddings)

        if sampling_str == "All samples":
            return np.arange(total_samples)

        elif sampling_str == "Random patients (all exams)":
            # Seleziona N pazienti casuali, prendi TUTTI i loro esami
            unique_patients = np.unique(patient_ids)
            num_patients = max(1, len(unique_patients) // 4)  # ~25% dei pazienti
            selected_patients = np.random.choice(unique_patients, num_patients, replace=False)
            indices = np.where(np.isin(patient_ids, selected_patients))[0]
            return indices

        else:
            # Numero di sample (1000, 2000, 5000, 10000)
            sample_size = int(sampling_str.split()[0])
            sample_size = min(sample_size, total_samples)
            indices = np.random.choice(total_samples, sample_size, replace=False)
            return indices

    def _compute_metrics(self, embeddings, patient_ids, sample_indices=None):
        """Calcola ROC-AUC e ranking accuracy"""
        # Usa campionamento se specificato
        if sample_indices is not None:
            embeddings = embeddings[sample_indices]
            patient_ids = patient_ids[sample_indices]

        # Crea coppie
        distances = []
        labels = []

        unique_patients = np.unique(patient_ids)

        for i in range(len(embeddings)):
            if self.stop_evaluation:
                break

            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                label = 1 if patient_ids[i] == patient_ids[j] else 0

                distances.append(dist)
                labels.append(label)

        distances = np.array(distances)
        labels = np.array(labels)

        # ROC-AUC (invertire distanze: più piccola = più simile)
        scores = -distances  # Negative per roc_auc (higher score = positive)
        roc_auc = roc_auc_score(labels, scores)

        # Ranking accuracy
        top1, top5, top10 = 0, 0, 0
        total = 0

        for i in range(len(embeddings)):
            if self.stop_evaluation:
                break

            query_emb = embeddings[i]
            query_patient = patient_ids[i]

            # Calcola distanze con tutti gli altri
            dists = [np.linalg.norm(query_emb - embeddings[j]) for j in range(len(embeddings))]
            sorted_indices = np.argsort(dists)

            # Skipping self (primo elemento è se stesso con distanza 0)
            ranking = patient_ids[sorted_indices[1:]]

            if ranking[0] == query_patient:
                top1 += 1
            if query_patient in ranking[:5]:
                top5 += 1
            if query_patient in ranking[:10]:
                top10 += 1

            total += 1

        top1_pct = (top1 / total) * 100 if total > 0 else 0
        top5_pct = (top5 / total) * 100 if total > 0 else 0
        top10_pct = (top10 / total) * 100 if total > 0 else 0

        return roc_auc, top1_pct, top5_pct, top10_pct

    def run_best_epoch_evaluation(self):
        """Esegue valutazione del modello sulla best epoch"""
        if self.current_run_path is None or self.results_json is None:
            self.statusBar().showMessage("❌ Nessuna run caricata")
            return

        best_epoch = self.results_json.get('best_epoch')
        if best_epoch is None:
            self.statusBar().showMessage("❌ Best epoch not found in results")
            return

        model_path = self.current_run_path / "models" / f"model_epoch_{best_epoch:04d}.pth"

        if not model_path.exists():
            self.statusBar().showMessage(f"❌ Model epoch {best_epoch} not found")
            return

        datasets = []
        if self.best_train_cb.isChecked():
            datasets.append(('train', 'data/ECG/train.csv'))
        if self.best_val_cb.isChecked():
            datasets.append(('val', 'data/ECG/val.csv'))
        if self.best_test_cb.isChecked():
            datasets.append(('test', 'data/ECG/test.csv'))

        if not datasets:
            self.statusBar().showMessage("❌ Seleziona almeno un dataset")
            return

        self.stop_evaluation = False
        self.best_eval_btn.setEnabled(False)
        self.best_stop_eval_btn.setEnabled(True)

        self.statusBar().showMessage(f"Computing evaluation for best epoch {best_epoch}...")
        self.best_epoch_results.setText("Computing...\n")

        try:
            # Carica encoder dal config
            from src.ecg_encoder import ECGEncoder

            config_file = self.current_run_path.parent.parent / "train_config_v2.yaml"
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Carica modello
            encoder = ECGEncoder(
                input_dim=config['encoder']['input_dim'],
                hidden_dims=config['encoder'].get('hidden_dims', [20]),
                embedding_dim=config['encoder']['embedding_dim'],
                dropout=config['encoder']['dropout'],
                normalize=config['encoder']['normalize']
            ).to(self.device)

            encoder.load_state_dict(torch.load(model_path, map_location=self.device))
            encoder.eval()

            sampling_str = self.best_sampling_combo.currentText()
            results_text = f"=== Best Epoch {best_epoch} Evaluation ===\n"
            results_text += f"Sampling: {sampling_str}\n\n"

            for dataset_name, csv_path in datasets:
                if self.stop_evaluation:
                    break

                csv_full_path = Path(csv_path) if Path(csv_path).is_absolute() else self.current_run_path.parent.parent / csv_path

                if not csv_full_path.exists():
                    results_text += f"\n❌ {dataset_name}: File not found ({csv_full_path})\n"
                    continue

                # Carica dataset
                df = pd.read_csv(csv_full_path)
                feature_cols = [
                    'VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                    'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset'
                ]
                features = df[feature_cols].values.astype(np.float32)
                patient_ids = df['PatientID'].values

                # Normalizza features
                scaler_mean = features.mean(axis=0)
                scaler_std = features.std(axis=0)
                features = (features - scaler_mean) / (scaler_std + 1e-8)

                # Calcola embeddings
                embeddings = self._get_embeddings(encoder, features)

                # Campionamento
                sample_indices = self._get_sample_indices(embeddings, patient_ids, sampling_str)

                # Calcola metriche
                roc_auc, top1, top5, top10 = self._compute_metrics(embeddings, patient_ids, sample_indices)

                results_text += f"\n{'='*40}\n"
                results_text += f"{dataset_name.upper()}\n"
                results_text += f"{'='*40}\n"
                results_text += f"ROC-AUC:       {roc_auc:.4f}\n"
                results_text += f"Ranking@1:     {top1:.2f}%\n"
                results_text += f"Ranking@5:     {top5:.2f}%\n"
                results_text += f"Ranking@10:    {top10:.2f}%\n"

            self.best_epoch_results.setText(results_text)
            if self.stop_evaluation:
                self.statusBar().showMessage("Evaluation stopped by user")
            else:
                self.statusBar().showMessage(f"✓ Evaluation complete (best epoch {best_epoch})")

        except Exception as e:
            self.best_epoch_results.setText(f"❌ Error: {str(e)}")
            self.statusBar().showMessage(f"❌ Error: {str(e)}")

        finally:
            self.best_eval_btn.setEnabled(True)
            self.best_stop_eval_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    visualizer = TrainingVisualizer()
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
