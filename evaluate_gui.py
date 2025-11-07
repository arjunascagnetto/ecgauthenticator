#!/usr/bin/env python3
"""Evaluation GUI per metric learning - Qt5."""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTabWidget, QLabel, QComboBox, QCheckBox, QGridLayout,
    QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))
from src.ecg_encoder import ECGEncoder
from src.ecg_metric_dataset import ECGPairDataset
from src.evaluation_utils import compute_batch_metrics
from src.extended_metrics import compute_extended_metrics_selective


class EvaluationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metric Learning Evaluation")
        self.setGeometry(100, 100, 1600, 900)

        self.run_dir = None
        self.history = None
        self.config = None
        self.encoder = None
        self.device = torch.device('cpu')
        self.test_dataset = None
        self.tabs_loaded = {}
        self.cached_embeddings_best = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Top: Load button
        top_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Run")
        self.load_btn.clicked.connect(self.load_run)
        self.run_label = QLabel("No run loaded")
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.run_label)
        layout.addLayout(top_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tab_training = QWidget()
        self.tab_validation = QWidget()
        self.tab_best = QWidget()
        self.tab_epoch = QWidget()
        self.tab_evolution = QWidget()

        self.tabs.addTab(self.tab_training, "Training")
        self.tabs.addTab(self.tab_validation, "Validation")
        self.tabs.addTab(self.tab_best, "Best Model")
        self.tabs.addTab(self.tab_epoch, "Epoch Model")
        self.tabs.addTab(self.tab_evolution, "Metrics Evolution")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tabs)
        main_widget.setLayout(layout)

        self.setup_training_tab()
        self.setup_validation_tab()
        self.setup_best_tab()
        self.setup_epoch_tab()
        self.setup_evolution_tab()

    def load_run(self):
        path = QFileDialog.getExistingDirectory(self, "Select Run Directory")
        if not path:
            return

        run_dir = Path(path)
        history_file = run_dir / 'history.csv'
        config_file = list(run_dir.glob('config_*.yaml'))
        results_file = run_dir / 'results.json'

        if not history_file.exists():
            self.run_label.setText("Invalid run directory")
            return

        self.run_dir = run_dir
        self.history = pd.read_csv(history_file)
        self.run_label.setText(f"Loaded: {run_dir.name}")

        with open(results_file) as f:
            self.results = json.load(f)

        # Load encoder
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)

        import yaml
        with open(config_file[0]) as f:
            self.config = yaml.safe_load(f)

        self.encoder = ECGEncoder(
            input_dim=self.config['encoder']['input_dim'],
            hidden_dims=self.config['encoder'].get('hidden_dims', [20]),
            embedding_dim=self.config['encoder']['embedding_dim'],
            dropout=self.config['encoder']['dropout'],
            normalize=self.config['encoder']['normalize']
        ).to(self.device)

        best_model_path = run_dir / 'models' / 'model_best.pth'
        self.encoder.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.encoder.eval()

        # Load test dataset
        test_csv = self.config['data'].get('test_csv', '/Users/arjuna/Progetti/siamese/data/ECG/test.csv')
        self.test_dataset = ECGPairDataset(test_csv, mining_strategy='random')

        # Reset lazy loading cache
        self.tabs_loaded = {}
        self.cached_embeddings_best = None

        # Plot only Training/Validation (fast)
        self.plot_training()
        self.plot_validation()

    def get_embeddings(self, features, batch_size=256, encoder=None):
        """Compute embeddings."""
        if encoder is None:
            encoder = self.encoder
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.from_numpy(features[i:i+batch_size]).float().to(self.device)
                embs = encoder(batch).cpu().numpy()
                all_embs.extend(embs)
        return np.array(all_embs)

    def on_tab_changed(self, index):
        """Lazy load tab content when activated."""
        tab_names = ["Training", "Validation", "Best Model", "Epoch Model", "Metrics Evolution"]
        tab_name = tab_names[index]

        if tab_name in self.tabs_loaded:
            return  # Already loaded

        if tab_name == "Best Model":
            self.plot_best()
            self.tabs_loaded["Best Model"] = True
        elif tab_name == "Epoch Model":
            self.refresh_epoch_combo()
            self.tabs_loaded["Epoch Model"] = True
        elif tab_name == "Metrics Evolution":
            self.plot_evolution()
            self.tabs_loaded["Metrics Evolution"] = True

    def setup_training_tab(self):
        layout = QGridLayout()
        self.fig_train = Figure(figsize=(14, 8))
        self.canvas_train = FigureCanvas(self.fig_train)
        layout.addWidget(self.canvas_train)
        self.tab_training.setLayout(layout)

    def setup_validation_tab(self):
        layout = QGridLayout()
        self.fig_val = Figure(figsize=(14, 8))
        self.canvas_val = FigureCanvas(self.fig_val)
        layout.addWidget(self.canvas_val)
        self.tab_validation.setLayout(layout)

    def setup_best_tab(self):
        layout = QVBoxLayout()

        # Metrics selection
        metrics_layout = QGridLayout()
        self.best_metrics_checks = {}
        for i, metric in enumerate(['ch', 'db', 'silhouette', 'nn_accuracy',
                                     'intra_inter_ratio', 'cluster_compactness', 'cluster_separation']):
            cb = QCheckBox(metric)
            cb.setChecked(metric in ['ch', 'db', 'intra_inter_ratio', 'cluster_compactness', 'cluster_separation'])
            self.best_metrics_checks[metric] = cb
            metrics_layout.addWidget(cb, i // 4, i % 4)

        layout.addLayout(metrics_layout)

        # Compute button
        self.best_compute_btn = QPushButton("Compute Metrics")
        self.best_compute_btn.clicked.connect(self.compute_best_metrics)
        layout.addWidget(self.best_compute_btn)

        # Progress bar + log
        self.best_progress = QProgressBar()
        self.best_progress.setVisible(False)
        layout.addWidget(self.best_progress)

        self.best_log = QTextEdit()
        self.best_log.setReadOnly(True)
        self.best_log.setMaximumHeight(100)
        layout.addWidget(self.best_log)

        # Results
        self.fig_best = Figure(figsize=(14, 6))
        self.canvas_best = FigureCanvas(self.fig_best)
        layout.addWidget(self.canvas_best)
        self.tab_best.setLayout(layout)

    def setup_epoch_tab(self):
        layout = QVBoxLayout()

        # Epoch selection
        epochs_label = QLabel("Select Epochs:")
        layout.addWidget(epochs_label)

        self.epoch_checks = {}
        self.epoch_select_all_btn = QPushButton("Select All / None")
        self.epoch_select_all_btn.clicked.connect(self.toggle_all_epochs)
        layout.addWidget(self.epoch_select_all_btn)

        self.epoch_checks_layout = QGridLayout()
        layout.addLayout(self.epoch_checks_layout)

        # Metrics selection
        metrics_label = QLabel("Select Metrics:")
        layout.addWidget(metrics_label)

        metrics_layout = QGridLayout()
        self.epoch_metrics_checks = {}
        for i, metric in enumerate(['ch', 'db', 'silhouette', 'nn_accuracy',
                                     'intra_inter_ratio', 'cluster_compactness', 'cluster_separation']):
            cb = QCheckBox(metric)
            cb.setChecked(metric in ['ch', 'db', 'intra_inter_ratio', 'cluster_compactness', 'cluster_separation'])
            self.epoch_metrics_checks[metric] = cb
            metrics_layout.addWidget(cb, i // 4, i % 4)

        layout.addLayout(metrics_layout)

        # Compute button
        self.epoch_compute_btn = QPushButton("Compute Metrics")
        self.epoch_compute_btn.clicked.connect(self.compute_epoch_metrics)
        layout.addWidget(self.epoch_compute_btn)

        # Progress bar + log
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setVisible(False)
        layout.addWidget(self.epoch_progress)

        self.epoch_log = QTextEdit()
        self.epoch_log.setReadOnly(True)
        self.epoch_log.setMaximumHeight(100)
        layout.addWidget(self.epoch_log)

        # Plot
        self.fig_epoch = Figure(figsize=(14, 6))
        self.canvas_epoch = FigureCanvas(self.fig_epoch)
        layout.addWidget(self.canvas_epoch)
        self.tab_epoch.setLayout(layout)

    def setup_evolution_tab(self):
        layout = QVBoxLayout()
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Metric:"))
        self.evo_metric = QComboBox()
        self.evo_metric.addItems(['ch', 'db'])
        self.evo_metric.currentIndexChanged.connect(self.on_evo_metric_changed)
        sel_layout.addWidget(self.evo_metric)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        self.fig_evo = Figure(figsize=(14, 6))
        self.canvas_evo = FigureCanvas(self.fig_evo)
        layout.addWidget(self.canvas_evo)
        self.tab_evolution.setLayout(layout)

    def plot_training(self):
        if self.history is None:
            return
        self.fig_train.clear()
        h = self.history
        ax = self.fig_train.subplots(2, 2)

        ax[0, 0].plot(h['epoch'], h['train_loss'], 'b-')
        ax[0, 0].set_title('Train Loss')
        ax[0, 0].grid(True)

        ax[0, 1].plot(h['epoch'], h['train_active_neg_pct'], 'g-')
        ax[0, 1].set_title('Active Negatives %')
        ax[0, 1].grid(True)

        mining_strat = h['mining_strategy'].unique()
        colors = {'random': 'r', 'semi-hard': 'orange', 'hard': 'purple'}
        for strat in mining_strat:
            mask = h['mining_strategy'] == strat
            ax[1, 0].scatter(h[mask]['epoch'], h[mask]['val_ch'], label=strat, color=colors.get(strat, 'gray'))
        ax[1, 0].set_title('Validation CH (by mining strategy)')
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        for strat in mining_strat:
            mask = h['mining_strategy'] == strat
            ax[1, 1].scatter(h[mask]['epoch'], h[mask]['val_db'], label=strat, color=colors.get(strat, 'gray'))
        ax[1, 1].set_title('Validation DB (by mining strategy)')
        ax[1, 1].legend()
        ax[1, 1].grid(True)

        self.fig_train.tight_layout()
        self.canvas_train.draw()

    def plot_validation(self):
        if self.history is None:
            return
        self.fig_val.clear()
        h = self.history
        ax = self.fig_val.subplots(1, 2)

        ax[0].plot(h['epoch'], h['val_ch'], 'b-o', markersize=4)
        ax[0].axvline(self.results['best_epoch'], color='r', linestyle='--', label=f"Best (ep {self.results['best_epoch']})")
        ax[0].set_title('Validation CH')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(h['epoch'], h['val_db'], 'g-o', markersize=4)
        ax[1].axvline(self.results['best_epoch'], color='r', linestyle='--', label=f"Best (ep {self.results['best_epoch']})")
        ax[1].set_title('Validation DB')
        ax[1].legend()
        ax[1].grid(True)

        self.fig_val.tight_layout()
        self.canvas_val.draw()

    def plot_best(self):
        if self.test_dataset is None:
            return
        # Placeholder - compute on demand

    def log_best(self, msg):
        """Log message to best tab."""
        self.best_log.append(msg)
        self.best_log.verticalScrollBar().setValue(self.best_log.verticalScrollBar().maximum())

    def compute_best_metrics(self):
        if self.test_dataset is None:
            return

        self.best_compute_btn.setText("Computing...")
        self.best_compute_btn.setEnabled(False)
        self.best_progress.setVisible(True)
        self.best_progress.setValue(0)
        self.best_log.clear()

        try:
            # Cache embeddings to avoid recomputation
            if self.cached_embeddings_best is None:
                self.log_best("Computing embeddings for best model...")
                self.best_progress.setValue(30)
                self.cached_embeddings_best = self.get_embeddings(self.test_dataset.features)
                self.log_best("✓ Embeddings computed")

            embs = self.cached_embeddings_best
            pids = self.test_dataset.patient_ids

            # Get selected metrics
            selected = [m for m, cb in self.best_metrics_checks.items() if cb.isChecked()]
            self.best_progress.setValue(60)

            self.log_best(f"Computing metrics: {', '.join(selected)}")
            batch_metrics = compute_batch_metrics(embs, pids)
            self.log_best("✓ Basic metrics computed")

            self.best_progress.setValue(75)
            ext_metrics = compute_extended_metrics_selective(embs, pids, selected)
            self.log_best("✓ Extended metrics computed")

            self.best_progress.setValue(90)

            # Build text
            text = f"Best Model Test Set Metrics (Epoch {self.results['best_epoch']}):\n\n"
            if 'ch' in selected:
                text += f"CH: {batch_metrics['ch']:.4f}\n"
            if 'db' in selected:
                text += f"DB: {batch_metrics['db']:.4f}\n"
            for metric in ['silhouette', 'nn_accuracy', 'intra_inter_ratio', 'cluster_compactness', 'cluster_separation']:
                if metric in selected and metric in ext_metrics:
                    text += f"{metric}: {ext_metrics[metric]:.4f}\n"

            self.fig_best.clear()
            ax = self.fig_best.add_subplot(111)
            ax.text(0.1, 0.5, text, fontfamily='monospace', fontsize=10, transform=ax.transAxes)
            ax.axis('off')
            self.canvas_best.draw()

            self.best_progress.setValue(100)
            self.log_best("✓ Done!")

        finally:
            self.best_compute_btn.setText("Compute Metrics")
            self.best_compute_btn.setEnabled(True)
            self.best_progress.setVisible(False)

    def refresh_epoch_combo(self):
        if self.history is None:
            return
        epochs = self.history['epoch'].tolist()

        # Clear old checkboxes
        for cb in self.epoch_checks.values():
            cb.deleteLater()
        self.epoch_checks.clear()

        # Add new checkboxes
        for i, epoch in enumerate(epochs):
            cb = QCheckBox(f"Epoch {epoch}")
            cb.setChecked(True)
            self.epoch_checks[epoch] = cb
            self.epoch_checks_layout.addWidget(cb, i // 5, i % 5)

    def toggle_all_epochs(self):
        """Toggle all epoch checkboxes."""
        if not self.epoch_checks:
            return
        all_checked = all(cb.isChecked() for cb in self.epoch_checks.values())
        for cb in self.epoch_checks.values():
            cb.setChecked(not all_checked)

    def log_epoch(self, msg):
        """Log message to epoch tab."""
        self.epoch_log.append(msg)
        self.epoch_log.verticalScrollBar().setValue(self.epoch_log.verticalScrollBar().maximum())

    def compute_epoch_metrics(self):
        if self.history is None or self.run_dir is None:
            self.log_epoch("Error: No run loaded")
            return

        # Get selected epochs
        selected_epochs = [e for e, cb in self.epoch_checks.items() if cb.isChecked()]
        if not selected_epochs:
            self.log_epoch("Error: No epochs selected")
            return

        # Get selected metrics
        selected_metrics = [m for m, cb in self.epoch_metrics_checks.items() if cb.isChecked()]
        if not selected_metrics:
            self.log_epoch("Error: No metrics selected")
            return

        self.epoch_compute_btn.setText("Computing...")
        self.epoch_compute_btn.setEnabled(False)
        self.epoch_progress.setVisible(True)
        self.epoch_progress.setValue(0)
        self.epoch_log.clear()

        try:
            results = {}  # epoch -> {metric -> value}
            pids = self.test_dataset.patient_ids
            total = len(selected_epochs)

            for idx, epoch in enumerate(selected_epochs):
                model_path = self.run_dir / 'models' / f'model_epoch_{epoch:04d}.pth'

                if not model_path.exists():
                    self.log_epoch(f"✗ Model for epoch {epoch} not found")
                    continue

                self.log_epoch(f"Loading epoch {epoch}...")
                encoder_tmp = ECGEncoder(
                    input_dim=self.config['encoder']['input_dim'],
                    hidden_dims=self.config['encoder'].get('hidden_dims', [20]),
                    embedding_dim=self.config['encoder']['embedding_dim'],
                    dropout=self.config['encoder']['dropout'],
                    normalize=self.config['encoder']['normalize']
                ).to(self.device)
                encoder_tmp.load_state_dict(torch.load(model_path, map_location=self.device))
                encoder_tmp.eval()

                embs = self.get_embeddings(self.test_dataset.features, encoder=encoder_tmp)
                batch_metrics = compute_batch_metrics(embs, pids)
                ext_metrics = compute_extended_metrics_selective(embs, pids, selected_metrics)

                results[epoch] = {}
                if 'ch' in selected_metrics:
                    results[epoch]['ch'] = batch_metrics['ch']
                if 'db' in selected_metrics:
                    results[epoch]['db'] = batch_metrics['db']
                results[epoch].update(ext_metrics)

                self.log_epoch(f"✓ Epoch {epoch} computed")
                self.epoch_progress.setValue(int((idx + 1) / total * 100))

            # Plot results
            self.log_epoch("Plotting results...")
            self.plot_epoch_evolution(results, selected_epochs, selected_metrics)
            self.log_epoch("✓ Done!")

        except Exception as e:
            self.log_epoch(f"Error: {str(e)}")

        finally:
            self.epoch_compute_btn.setText("Compute Metrics")
            self.epoch_compute_btn.setEnabled(True)
            self.epoch_progress.setVisible(False)

    def plot_epoch_evolution(self, results, epochs, metrics):
        """Plot evolution of metrics across epochs."""
        self.fig_epoch.clear()
        num_metrics = len(metrics)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols

        for idx, metric in enumerate(metrics):
            ax = self.fig_epoch.add_subplot(rows, cols, idx + 1)

            values = [results[e].get(metric, 0) for e in epochs]
            ax.plot(epochs, values, 'o-', markersize=6, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Evolution')
            ax.grid(True, alpha=0.3)

        self.fig_epoch.tight_layout()
        self.canvas_epoch.draw()

    def on_evo_metric_changed(self):
        """Redraw evolution when metric dropdown changes."""
        self.plot_evolution()

    def plot_evolution(self):
        if self.history is None:
            return
        self.fig_evo.clear()
        ax = self.fig_evo.add_subplot(111)

        metric = self.evo_metric.currentText()
        col = f'val_{metric}'

        ax.plot(self.history['epoch'], self.history[col], 'o-', markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Evolution')
        ax.grid(True)

        self.fig_evo.tight_layout()
        self.canvas_evo.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EvaluationGUI()
    gui.show()
    sys.exit(app.exec_())
