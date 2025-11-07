import sys
import os
import glob
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QFileDialog, QMessageBox, QGroupBox, QSpinBox, QRadioButton,
                             QButtonGroup, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class EmbeddingViewer3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Embedding Evolution Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Dati
        self.train_files = []
        self.test_files = []
        self.embeddings_3d = []  # Lista di embeddings ridotti a 3D per ogni epoca
        self.labels_list = []     # Lista di labels per ogni epoca
        self.epochs_list = []     # Lista numeri epoca
        self.metrics_list = []    # Lista di metriche per ogni epoca
        self.current_epoch_idx = 0
        self.dataset_mode = 'train'  # 'train', 'test', o 'both'

        # Animazione
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.animation_speed = 500  # millisecondi

        # Crea interfaccia
        self.init_ui()

        # Carica automaticamente gli embeddings
        self.auto_load_embeddings()

    def init_ui(self):
        """Crea l'interfaccia grafica"""

        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Barra controlli superiore ---
        control_layout = QHBoxLayout()

        load_btn = QPushButton("📂 Load Embeddings")
        load_btn.clicked.connect(self.load_embeddings)
        load_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        control_layout.addWidget(load_btn)

        self.file_label = QLabel("No embeddings loaded")
        self.file_label.setStyleSheet("padding: 8px;")
        control_layout.addWidget(self.file_label)

        # Radio buttons per selezionare dataset
        dataset_label = QLabel("Dataset:")
        dataset_label.setStyleSheet("font-weight: bold; padding: 4px;")
        control_layout.addWidget(dataset_label)

        self.dataset_button_group = QButtonGroup()

        self.train_radio = QRadioButton("Train")
        self.train_radio.setChecked(True)
        self.train_radio.toggled.connect(lambda: self.set_dataset_mode('train'))
        self.dataset_button_group.addButton(self.train_radio)
        control_layout.addWidget(self.train_radio)

        self.test_radio = QRadioButton("Test")
        self.test_radio.toggled.connect(lambda: self.set_dataset_mode('test'))
        self.dataset_button_group.addButton(self.test_radio)
        control_layout.addWidget(self.test_radio)

        process_btn = QPushButton("🔄 Apply t-SNE")
        process_btn.clicked.connect(self.process_embeddings)
        process_btn.setStyleSheet("padding: 8px;")
        control_layout.addWidget(process_btn)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # --- Plot 3D ---
        plot_group = QGroupBox("📊 3D Embedding Space Evolution")
        plot_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        plot_layout = QVBoxLayout()

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        plot_layout.addWidget(self.canvas)

        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)

        # --- Controlli animazione ---
        animation_group = QGroupBox("🎬 Animation Controls")
        animation_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11pt; }")
        animation_layout = QVBoxLayout()

        # Riga 1: Play/Pause e epoch info
        buttons_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        buttons_layout.addWidget(self.play_btn)

        self.prev_btn = QPushButton("⏮️ Previous")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        buttons_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("⏭️ Next")
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        buttons_layout.addWidget(self.next_btn)

        self.epoch_label = QLabel("Epoch: -")
        self.epoch_label.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 8px;")
        buttons_layout.addWidget(self.epoch_label)

        buttons_layout.addStretch()
        animation_layout.addLayout(buttons_layout)

        # Riga 2: Slider epoca
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Epoch:"))

        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setEnabled(False)
        self.epoch_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.epoch_slider)

        animation_layout.addLayout(slider_layout)

        # Riga 3: Velocità animazione
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed (ms):"))

        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(100, 2000)
        self.speed_spinbox.setValue(500)
        self.speed_spinbox.setSingleStep(100)
        self.speed_spinbox.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_spinbox)

        speed_layout.addStretch()
        animation_layout.addLayout(speed_layout)

        animation_group.setLayout(animation_layout)
        main_layout.addWidget(animation_group)

        # --- Box metriche ---
        metrics_group = QGroupBox("📈 Embedding Quality Metrics")
        metrics_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11pt; }")
        metrics_layout = QVBoxLayout()

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("font-family: 'Courier'; font-size: 10pt;")
        self.metrics_text.setMaximumHeight(100)
        self.metrics_text.setText("No metrics available. Load and process embeddings first.")
        metrics_layout.addWidget(self.metrics_text)

        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

    def set_dataset_mode(self, mode):
        """Cambia il dataset mode (train/test)"""
        self.dataset_mode = mode
        # Resetta embeddings 3D per forzare ricalcolo
        if self.train_files or self.test_files:
            QMessageBox.information(self, "Dataset Changed",
                                   f"Dataset switched to {mode.upper()}.\nClick 'Apply t-SNE' to reprocess.")

    def auto_load_embeddings(self):
        """Carica automaticamente tutti gli embedding files trovati"""
        train_files = sorted(glob.glob('logs/embeddings_train_epoch_*_*.npz'))
        test_files = sorted(glob.glob('logs/embeddings_test_epoch_*_*.npz'))

        # Fallback: cerca vecchi file senza train/test nel nome
        if not train_files and not test_files:
            old_files = sorted(glob.glob('logs/embeddings_epoch_*_*.npz'))
            if old_files:
                self.train_files = old_files  # Tratta come train per compatibilità
                self.file_label.setText(f"Found {len(old_files)} old-format files (treated as train)")
                QMessageBox.information(self, "Auto-load",
                                       f"Found {len(old_files)} old-format embedding files.\n"
                                       f"Click 'Apply t-SNE' to process them.")
            return

        # Raggruppa per timestamp
        train_timestamps = {}
        test_timestamps = {}

        for f in train_files:
            parts = os.path.basename(f).split('_')
            if len(parts) >= 5:
                timestamp = '_'.join(parts[4:]).replace('.npz', '')
                if timestamp not in train_timestamps:
                    train_timestamps[timestamp] = []
                train_timestamps[timestamp].append(f)

        for f in test_files:
            parts = os.path.basename(f).split('_')
            if len(parts) >= 5:
                timestamp = '_'.join(parts[4:]).replace('.npz', '')
                if timestamp not in test_timestamps:
                    test_timestamps[timestamp] = []
                test_timestamps[timestamp].append(f)

        # Prendi l'ultimo timestamp
        if train_timestamps or test_timestamps:
            all_timestamps = set(train_timestamps.keys()) | set(test_timestamps.keys())
            latest_timestamp = sorted(all_timestamps)[-1]

            self.train_files = sorted(train_timestamps.get(latest_timestamp, []))
            self.test_files = sorted(test_timestamps.get(latest_timestamp, []))

            msg = f"Timestamp: {latest_timestamp}\n"
            msg += f"Train files: {len(self.train_files)}\n"
            msg += f"Test files: {len(self.test_files)}"

            self.file_label.setText(f"Found {len(self.train_files)} train + {len(self.test_files)} test files")
            QMessageBox.information(self, "Auto-load", f"{msg}\n\nClick 'Apply t-SNE' to process.")

    def load_embeddings(self):
        """Apre dialog per selezionare embedding files"""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select embedding files",
            "logs",
            "NumPy files (*.npz);;All files (*.*)"
        )
        if filenames:
            self.embedding_files = sorted(filenames)
            self.file_label.setText(f"Loaded {len(self.embedding_files)} files")

    def process_embeddings(self):
        """Applica t-SNE a tutti gli embeddings e prepara la visualizzazione"""
        # Seleziona i file in base al dataset_mode
        if self.dataset_mode == 'train':
            files_to_process = self.train_files
        elif self.dataset_mode == 'test':
            files_to_process = self.test_files
        else:
            QMessageBox.warning(self, "Warning", "Invalid dataset mode!")
            return

        if not files_to_process:
            QMessageBox.warning(self, "Warning", f"No {self.dataset_mode} embedding files loaded!")
            return

        try:
            QMessageBox.information(self, "Processing",
                                   f"Applying t-SNE to {self.dataset_mode.upper()} embeddings. This may take a minute...")

            self.embeddings_3d = []
            self.labels_list = []
            self.epochs_list = []
            self.metrics_list = []

            # Carica tutti gli embeddings
            all_embeddings = []
            for f in files_to_process:
                data = np.load(f)
                all_embeddings.append(data['embeddings'])
                self.labels_list.append(data['labels'])
                self.epochs_list.append(int(data['epoch']))

                # Carica metriche se disponibili
                metrics = {}
                if 'db_index' in data:
                    metrics['db_index'] = float(data['db_index'])
                    metrics['ch_score'] = float(data['ch_score'])
                    metrics['map'] = float(data['map'])
                    metrics['nmi'] = float(data['nmi'])
                self.metrics_list.append(metrics)

            # Concatena tutti gli embeddings per applicare t-SNE una volta sola
            # Questo garantisce uno spazio embedding coerente tra le epoche
            concatenated = np.vstack(all_embeddings)

            # Applica t-SNE
            print(f"Applying t-SNE to {self.dataset_mode} embeddings...")
            tsne = TSNE(n_components=3, random_state=0, perplexity=30)
            embeddings_3d_all = tsne.fit_transform(concatenated)

            # Dividi di nuovo per epoca
            start_idx = 0
            for emb in all_embeddings:
                n_samples = len(emb)
                self.embeddings_3d.append(embeddings_3d_all[start_idx:start_idx + n_samples])
                start_idx += n_samples

            # Abilita controlli
            self.play_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.epoch_slider.setEnabled(True)
            self.epoch_slider.setRange(0, len(self.embeddings_3d) - 1)
            self.epoch_slider.setValue(0)

            # Mostra prima epoca
            self.current_epoch_idx = 0
            self.plot_epoch()

            QMessageBox.information(self, "Success", f"t-SNE processing completed for {self.dataset_mode.upper()} data!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process embeddings:\n{str(e)}")

    def plot_epoch(self):
        """Plotta gli embeddings per l'epoca corrente"""
        if not self.embeddings_3d:
            return

        # Clear della figura e ricrea l'asse 3D per evitare problemi con colorbar
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d')

        embeddings_3d = self.embeddings_3d[self.current_epoch_idx]
        labels = self.labels_list[self.current_epoch_idx]
        epoch_num = self.epochs_list[self.current_epoch_idx]

        # Plotta punti colorati per classe
        scatter = self.ax.scatter(embeddings_3d[:, 0],
                                 embeddings_3d[:, 1],
                                 embeddings_3d[:, 2],
                                 c=labels,
                                 cmap='tab10',
                                 s=50,
                                 alpha=0.7,
                                 edgecolors='k',
                                 linewidths=0.5)

        self.ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
        self.ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
        self.ax.set_zlabel('t-SNE Dimension 3', fontweight='bold')
        self.ax.set_title(f'Embedding Space ({self.dataset_mode.upper()}) - Epoch {epoch_num}',
                         fontweight='bold', fontsize=14)

        # Aggiungi colorbar
        colorbar = self.figure.colorbar(scatter, ax=self.ax, pad=0.1, shrink=0.8)
        colorbar.set_label('Digit Class', fontweight='bold')

        # Aggiorna label epoca
        self.epoch_label.setText(f"Epoch: {epoch_num} ({self.current_epoch_idx + 1}/{len(self.embeddings_3d)})")

        # Aggiorna metriche
        if self.current_epoch_idx < len(self.metrics_list) and self.metrics_list[self.current_epoch_idx]:
            metrics = self.metrics_list[self.current_epoch_idx]
            metrics_text = f"Epoch {epoch_num} - {self.dataset_mode.upper()} Dataset Metrics:\n"
            metrics_text += f"  Davies-Bouldin Index: {metrics['db_index']:.4f} (↓ lower is better)\n"
            metrics_text += f"  Calinski-Harabasz Score: {metrics['ch_score']:.2f} (↑ higher is better)\n"
            metrics_text += f"  Mean Average Precision (mAP): {metrics['map']:.4f}\n"
            metrics_text += f"  Normalized Mutual Information (NMI): {metrics['nmi']:.4f}"
            self.metrics_text.setText(metrics_text)
        else:
            self.metrics_text.setText("No metrics available for this epoch.")

        self.canvas.draw()

    def toggle_play(self):
        """Toggle animazione play/pause"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶️ Play")
        else:
            self.timer.start(self.animation_speed)
            self.is_playing = True
            self.play_btn.setText("⏸️ Pause")

    def next_frame(self):
        """Va al frame successivo"""
        if not self.embeddings_3d:
            return

        self.current_epoch_idx = (self.current_epoch_idx + 1) % len(self.embeddings_3d)
        self.epoch_slider.setValue(self.current_epoch_idx)
        self.plot_epoch()

    def prev_frame(self):
        """Va al frame precedente"""
        if not self.embeddings_3d:
            return

        self.current_epoch_idx = (self.current_epoch_idx - 1) % len(self.embeddings_3d)
        self.epoch_slider.setValue(self.current_epoch_idx)
        self.plot_epoch()

    def on_slider_changed(self, value):
        """Callback slider epoca"""
        if not self.embeddings_3d:
            return

        self.current_epoch_idx = value
        self.plot_epoch()

    def on_speed_changed(self, value):
        """Callback cambio velocità"""
        self.animation_speed = value
        if self.is_playing:
            self.timer.setInterval(self.animation_speed)


def main():
    app = QApplication(sys.argv)
    viewer = EmbeddingViewer3D()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
