#!/usr/bin/env python3
"""
Run Viewer - Visualizzatore Interattivo per Training Runs
Scansiona automaticamente la cartella runs/ e mostra metriche, grafici e configurazioni.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QTabWidget, QPushButton, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QTableWidget, QTableWidgetItem,
    QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RunInfo:
    """Container per informazioni run"""
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.timestamp = None
        self.epochs = 0
        self.best_ch = 0.0
        self.best_epoch = 0
        self.status = "unknown"
        self.has_history = False
        self.has_batch_metrics = False
        self.has_config = False
        self.has_results = False

        self._load_metadata()

    def _load_metadata(self):
        """Carica metadata dalla run"""
        # Estrai timestamp dal nome
        try:
            self.timestamp = self.name.replace("run_", "")
        except:
            pass

        # Controlla file esistenti
        logs_dir = self.path / "logs"
        self.has_history = (logs_dir / "train_history.csv").exists()
        self.has_batch_metrics = (logs_dir / "batch_metrics_train.csv").exists() and \
                                 (logs_dir / "batch_metrics_val.csv").exists()
        self.has_config = list(self.path.glob("config_*.yaml")) != []
        self.has_results = (self.path / "results" / "metrics.json").exists()

        # Carica metriche
        if self.has_history:
            try:
                df = pd.read_csv(logs_dir / "train_history.csv")
                self.epochs = len(df)
                if 'val_ch_score' in df.columns:
                    self.best_ch = df['val_ch_score'].max()
                    self.best_epoch = df[df['val_ch_score'] == self.best_ch]['epoch'].values[0]
            except Exception as e:
                print(f"Error loading history: {e}")

        # Status
        if self.has_results:
            self.status = "completed"
        elif self.has_history and self.epochs > 0:
            self.status = "interrupted"
        else:
            self.status = "incomplete"

    def __str__(self):
        status_icon = {"completed": "✓", "interrupted": "⚠", "incomplete": "✗"}.get(self.status, "?")
        return f"{status_icon} {self.name} ({self.epochs} ep, CH={self.best_ch:.2f})"


class MplCanvas(FigureCanvas):
    """Canvas matplotlib per PyQt5"""
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class RunListPanel(QWidget):
    """Panel sinistra con lista run"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs: List[RunInfo] = []
        self.selected_run: Optional[RunInfo] = None
        self.on_selection_changed = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Label
        title = QLabel("Available Runs")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        layout.addWidget(title)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Buttons
        button_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_runs)
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.clicked.connect(self._open_runs_folder)

        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.open_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_runs(self, base_dir: Path):
        """Carica lista run da cartella"""
        self.base_dir = base_dir
        self.refresh_runs()

    def refresh_runs(self):
        """Scansiona e aggiorna lista run"""
        if not hasattr(self, 'base_dir'):
            return

        self.list_widget.clear()
        self.runs = []

        if self.base_dir.exists():
            for run_dir in sorted(self.base_dir.glob("run_*"), reverse=True):
                if run_dir.is_dir():
                    run_info = RunInfo(run_dir)
                    self.runs.append(run_info)

                    item = QListWidgetItem(str(run_info))
                    # Colora in base a status
                    if run_info.status == "completed":
                        item.setBackground(QColor(200, 255, 200))
                    elif run_info.status == "interrupted":
                        item.setBackground(QColor(255, 255, 200))
                    else:
                        item.setBackground(QColor(255, 200, 200))

                    self.list_widget.addItem(item)

    def _on_selection_changed(self):
        """Handler selezione"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.runs):
            self.selected_run = self.runs[current_row]
            if self.on_selection_changed:
                self.on_selection_changed(self.selected_run)

    def _open_runs_folder(self):
        """Apri cartella run nel file manager"""
        if hasattr(self, 'base_dir') and self.base_dir.exists():
            import subprocess
            subprocess.Popen(['open', str(self.base_dir)])


class EpochMetricsTab(QWidget):
    """Tab 1: Metriche per epoca"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_data = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Canvas
        self.canvas = MplCanvas(self, width=14, height=8, dpi=100)
        layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Figure")
        self.save_btn.clicked.connect(self._save_figure)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def update_data(self, run_info: RunInfo):
        """Aggiorna grafici"""
        try:
            history_path = run_info.path / "logs" / "train_history.csv"
            if not history_path.exists():
                return

            self.df = pd.read_csv(history_path)
            self._plot_metrics()
        except Exception as e:
            print(f"Error loading epoch metrics: {e}")

    def _plot_metrics(self):
        """Plot metriche per epoca (5 grafici su griglia 3x2)"""
        self.canvas.fig.clear()
        axes = self.canvas.fig.subplots(3, 2)

        df = self.df

        # [0,0] - Loss Train
        axes[0, 0].plot(df['epoch'], df['train_loss_mean'], 'o-', color='blue', linewidth=2)
        axes[0, 0].fill_between(df['epoch'],
                               df['train_loss_mean'] - df['train_loss_std'],
                               df['train_loss_mean'] + df['train_loss_std'],
                               alpha=0.2, color='blue')
        axes[0, 0].set_title('Loss (Train)', fontweight='bold', fontsize=11)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # [0,1] - VUOTO
        axes[0, 1].axis('off')

        # [1,0] - DB + CH (Train, doppio asse)
        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        l1 = ax1.plot(df['epoch'], df['train_db_mean'], 'o-', color='purple', linewidth=2, label='DB (Train)')
        l2 = ax2.plot(df['epoch'], df['train_ch_mean'], 's-', color='green', linewidth=2, label='CH (Train)')
        ax1.set_title('DB + CH (Train)', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('DB Index', color='purple')
        ax2.set_ylabel('CH Score', color='green')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax1.grid(True, alpha=0.3)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # [1,1] - DB + CH (Val, doppio asse)
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        l1 = ax1.plot(df['epoch'], df['val_db_index'], 'o-', color='purple', linewidth=2, label='DB (Val)')
        l2 = ax2.plot(df['epoch'], df['val_ch_score'], 's-', color='green', linewidth=2, label='CH (Val)')
        ax1.set_title('DB + CH (Val)', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('DB Index', color='purple')
        ax2.set_ylabel('CH Score', color='green')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax1.grid(True, alpha=0.3)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # [2,0] - d_intra + d_inter (Train)
        axes[2, 0].plot(df['epoch'], df['train_d_intra_mean'], 'o-', color='cyan', linewidth=2, label='d_intra')
        axes[2, 0].plot(df['epoch'], df['train_d_inter_mean'], 's-', color='magenta', linewidth=2, label='d_inter')
        axes[2, 0].fill_between(df['epoch'],
                               df['train_d_intra_mean'] - df['train_d_intra_std'],
                               df['train_d_intra_mean'] + df['train_d_intra_std'],
                               alpha=0.1, color='cyan')
        axes[2, 0].fill_between(df['epoch'],
                               df['train_d_inter_mean'] - df['train_d_inter_std'],
                               df['train_d_inter_mean'] + df['train_d_inter_std'],
                               alpha=0.1, color='magenta')
        axes[2, 0].set_title('d_intra + d_inter (Train)', fontweight='bold', fontsize=11)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Distance')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()

        # [2,1] - d_intra + d_inter (Val)
        axes[2, 1].plot(df['epoch'], df['val_d_intra_mean'], 'o-', color='cyan', linewidth=2, label='d_intra')
        axes[2, 1].plot(df['epoch'], df['val_d_inter_mean'], 's-', color='magenta', linewidth=2, label='d_inter')
        axes[2, 1].fill_between(df['epoch'],
                               df['val_d_intra_mean'] - df['val_d_intra_std'],
                               df['val_d_intra_mean'] + df['val_d_intra_std'],
                               alpha=0.1, color='cyan')
        axes[2, 1].fill_between(df['epoch'],
                               df['val_d_inter_mean'] - df['val_d_inter_std'],
                               df['val_d_inter_mean'] + df['val_d_inter_std'],
                               alpha=0.1, color='magenta')
        axes[2, 1].set_title('d_intra + d_inter (Val)', fontweight='bold', fontsize=11)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Distance')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _save_figure(self):
        """Salva figura"""
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Saved to {path}")


class BatchMetricsTab(QWidget):
    """Tab 2: Metriche per batch"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_data = None
        self.df_batch_train = None
        self.df_batch_val = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Epoch selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Epoch:"))
        self.epoch_combo = QComboBox()
        self.epoch_combo.currentIndexChanged.connect(self._on_epoch_changed)
        selector_layout.addWidget(self.epoch_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        # Canvas
        self.canvas = MplCanvas(self, width=14, height=8, dpi=100)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def update_data(self, run_info: RunInfo):
        """Aggiorna dati batch"""
        try:
            logs_dir = run_info.path / "logs"
            self.df_batch_train = pd.read_csv(logs_dir / "batch_metrics_train.csv")
            self.df_batch_val = pd.read_csv(logs_dir / "batch_metrics_val.csv")

            # Popola epoch selector
            epochs = sorted(self.df_batch_train['epoch'].unique())
            self.epoch_combo.blockSignals(True)
            self.epoch_combo.clear()
            for epoch in epochs:
                self.epoch_combo.addItem(f"Epoch {epoch}", epoch)
            self.epoch_combo.blockSignals(False)

            # Plot primo epoch
            if len(epochs) > 0:
                self.epoch_combo.setCurrentIndex(0)
        except Exception as e:
            print(f"Error loading batch metrics: {e}")

    def _on_epoch_changed(self):
        """Handler cambio epoca"""
        epoch = self.epoch_combo.currentData()
        if epoch is not None:
            self._plot_batch_metrics(epoch)

    def _plot_batch_metrics(self, epoch: int):
        """Plot batch metrics per epoca (5 grafici su griglia 3x2)"""
        self.canvas.fig.clear()
        axes = self.canvas.fig.subplots(3, 2)

        df_train = self.df_batch_train[self.df_batch_train['epoch'] == epoch]
        df_val = self.df_batch_val[self.df_batch_val['epoch'] == epoch]

        if len(df_train) == 0 and len(df_val) == 0:
            axes[0, 0].text(0.5, 0.5, "No data for this epoch", ha='center', va='center')
            self.canvas.draw()
            return

        # [0,0] - Train Loss
        if len(df_train) > 0 and 'loss' in df_train.columns:
            axes[0, 0].bar(df_train['batch_idx'], df_train['loss'], color='blue', alpha=0.7)
            axes[0, 0].set_title(f'Loss (Train) - Epoch {epoch}', fontweight='bold', fontsize=11)
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)

        # [0,1] - VUOTO
        axes[0, 1].axis('off')

        # [1,0] - DB + CH (Train, doppio asse)
        if len(df_train) > 0 and 'db' in df_train.columns and 'ch' in df_train.columns:
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            l1 = ax1.plot(df_train['batch_idx'], df_train['db'], 'o-', color='purple', linewidth=2, label='DB (Train)')
            l2 = ax2.plot(df_train['batch_idx'], df_train['ch'], 's-', color='green', linewidth=2, label='CH (Train)')
            ax1.set_title(f'DB + CH (Train) - Epoch {epoch}', fontweight='bold', fontsize=11)
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('DB Index', color='purple')
            ax2.set_ylabel('CH Score', color='green')
            ax1.tick_params(axis='y', labelcolor='purple')
            ax2.tick_params(axis='y', labelcolor='green')
            ax1.grid(True, alpha=0.3)
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=9)

        # [1,1] - DB + CH (Val, doppio asse)
        if len(df_val) > 0 and 'db' in df_val.columns and 'ch' in df_val.columns:
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            l1 = ax1.plot(df_val['batch_idx'], df_val['db'], 'o-', color='purple', linewidth=2, label='DB (Val)')
            l2 = ax2.plot(df_val['batch_idx'], df_val['ch'], 's-', color='green', linewidth=2, label='CH (Val)')
            ax1.set_title(f'DB + CH (Val) - Epoch {epoch}', fontweight='bold', fontsize=11)
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('DB Index', color='purple')
            ax2.set_ylabel('CH Score', color='green')
            ax1.tick_params(axis='y', labelcolor='purple')
            ax2.tick_params(axis='y', labelcolor='green')
            ax1.grid(True, alpha=0.3)
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=9)

        # [2,0] - d_intra + d_inter (Train)
        if len(df_train) > 0 and 'd_intra' in df_train.columns and 'd_inter' in df_train.columns:
            axes[2, 0].plot(df_train['batch_idx'], df_train['d_intra'], 'o-', color='cyan', linewidth=2, label='d_intra')
            axes[2, 0].plot(df_train['batch_idx'], df_train['d_inter'], 's-', color='magenta', linewidth=2, label='d_inter')
            axes[2, 0].set_title(f'd_intra + d_inter (Train) - Epoch {epoch}', fontweight='bold', fontsize=11)
            axes[2, 0].set_xlabel('Batch')
            axes[2, 0].set_ylabel('Distance')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()

        # [2,1] - d_intra + d_inter (Val)
        if len(df_val) > 0 and 'd_intra' in df_val.columns and 'd_inter' in df_val.columns:
            axes[2, 1].plot(df_val['batch_idx'], df_val['d_intra'], 'o-', color='cyan', linewidth=2, label='d_intra')
            axes[2, 1].plot(df_val['batch_idx'], df_val['d_inter'], 's-', color='magenta', linewidth=2, label='d_inter')
            axes[2, 1].set_title(f'd_intra + d_inter (Val) - Epoch {epoch}', fontweight='bold', fontsize=11)
            axes[2, 1].set_xlabel('Batch')
            axes[2, 1].set_ylabel('Distance')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ConfigTab(QWidget):
    """Tab 3: Configurazione YAML"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Text display
        from PyQt5.QtWidgets import QTextEdit
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def update_data(self, run_info: RunInfo):
        """Aggiorna config display"""
        try:
            config_files = list(run_info.path.glob("config_*.yaml"))
            if config_files:
                with open(config_files[0], 'r') as f:
                    config_text = f.read()
                self.text_edit.setText(config_text)
        except Exception as e:
            self.text_edit.setText(f"Error loading config: {e}")


class StatisticsTab(QWidget):
    """Tab 4: Statistiche"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Summary table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.summary_table)

        self.setLayout(layout)

    def update_data(self, run_info: RunInfo):
        """Aggiorna statistiche"""
        try:
            self.summary_table.setRowCount(0)

            metrics = [
                ("Run Name", run_info.name),
                ("Status", run_info.status),
                ("Total Epochs", str(run_info.epochs)),
                ("Best Epoch", str(run_info.best_epoch)),
                ("Best CH (Val)", f"{run_info.best_ch:.4f}"),
            ]

            # Carica test metrics
            results_file = run_info.path / "results" / "metrics.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    test_metrics = results.get('test_metrics', {})
                    if 'calinski_harabasz' in test_metrics:
                        metrics.append(("Test CH", f"{test_metrics['calinski_harabasz']:.4f}"))
                    if 'davies_bouldin' in test_metrics:
                        metrics.append(("Test DB", f"{test_metrics['davies_bouldin']:.4f}"))
                    if 'silhouette' in test_metrics:
                        metrics.append(("Test Silhouette", f"{test_metrics['silhouette']:.4f}"))

            # Popola tabella
            for i, (metric, value) in enumerate(metrics):
                self.summary_table.insertRow(i)
                self.summary_table.setItem(i, 0, QTableWidgetItem(metric))
                self.summary_table.setItem(i, 1, QTableWidgetItem(str(value)))

        except Exception as e:
            print(f"Error loading statistics: {e}")


class RunViewer(QMainWindow):
    """Finestra principale"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Run Viewer - ECG Metric Learning")
        self.setGeometry(100, 100, 1600, 900)

        # Base directory
        self.base_dir = Path("/Users/arjuna/Progetti/siamese/runs")

        self.init_ui()
        self.init_runs()

    def init_ui(self):
        """Crea UI"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel: Run list
        self.run_list = RunListPanel()
        self.run_list.on_selection_changed = self._on_run_selected
        main_layout.addWidget(self.run_list, 1)

        # Right panel: Tabs
        right_layout = QVBoxLayout()

        # Run info label
        self.current_run_label = QLabel("No run selected")
        self.current_run_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        right_layout.addWidget(self.current_run_label)

        # Tab widget
        self.tabs = QTabWidget()
        self.epoch_metrics_tab = EpochMetricsTab()
        self.batch_metrics_tab = BatchMetricsTab()
        self.config_tab = ConfigTab()
        self.stats_tab = StatisticsTab()

        self.tabs.addTab(self.epoch_metrics_tab, "Epoch Metrics")
        self.tabs.addTab(self.batch_metrics_tab, "Batch Metrics")
        self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.stats_tab, "Statistics")

        right_layout.addWidget(self.tabs)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def init_runs(self):
        """Inizializza scansione run"""
        self.run_list.load_runs(self.base_dir)

    def _on_run_selected(self, run_info: RunInfo):
        """Handler selezione run"""
        self.current_run_label.setText(f"Selected: {run_info.name} | Status: {run_info.status}")

        # Aggiorna tutti i tab
        self.epoch_metrics_tab.update_data(run_info)
        self.batch_metrics_tab.update_data(run_info)
        self.config_tab.update_data(run_info)
        self.stats_tab.update_data(run_info)


def main():
    app = QApplication(sys.argv)
    viewer = RunViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
