"""
Visualizer per training runs - PyQt5 GUI
Legge history.csv da runs_v2 e plotta metriche di training
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QTabWidget, QLabel, QStatusBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class TrainingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Training Run Visualizer")
        self.setGeometry(100, 100, 1400, 900)

        self.runs_dir = Path("runs_v2")
        self.current_run = None
        self.history_df = None

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


def main():
    app = QApplication(sys.argv)
    visualizer = TrainingVisualizer()
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
