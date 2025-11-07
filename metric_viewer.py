import sys
import os
import glob
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QTextEdit, QFileDialog, QMessageBox, QGroupBox,
                             QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MetricViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Siamese Network Metric Viewer")
        self.setGeometry(100, 100, 1600, 1000)

        # Dati
        self.df = None
        self.current_csv = None
        self.test_results = None

        # Crea interfaccia
        self.init_ui()

        # Carica automaticamente l'ultimo file CSV se esiste
        self.auto_load_latest()

    def init_ui(self):
        """Crea l'interfaccia grafica con tab"""

        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Barra controlli superiore ---
        control_layout = QHBoxLayout()

        load_btn = QPushButton("📂 Load CSV")
        load_btn.clicked.connect(self.load_csv)
        load_btn.setStyleSheet("font-weight: bold; padding: 10px; font-size: 11pt;")
        control_layout.addWidget(load_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("padding: 10px; font-size: 11pt;")
        control_layout.addWidget(self.file_label)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        refresh_btn.setStyleSheet("padding: 10px; font-size: 11pt;")
        control_layout.addWidget(refresh_btn)

        export_btn = QPushButton("💾 Export Plots")
        export_btn.clicked.connect(self.export_plots)
        export_btn.setStyleSheet("padding: 10px; font-size: 11pt;")
        control_layout.addWidget(export_btn)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4a90e2;
                color: white;
            }
        """)

        # TAB 1: Epoch Metrics
        self.create_epoch_tab()

        # TAB 2: Batch Metrics
        self.create_batch_tab()

        # TAB 3: Results & Info
        self.create_results_tab()

        main_layout.addWidget(self.tab_widget)

    def create_epoch_tab(self):
        """Crea il tab per le metriche per epoca"""
        epoch_tab = QWidget()
        epoch_layout = QVBoxLayout(epoch_tab)

        # Titolo
        title_label = QLabel("📊 Training Metrics Evolution per Epoch")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        epoch_layout.addWidget(title_label)

        # Canvas per grafici
        self.epoch_figure = Figure(figsize=(14, 10), dpi=100)
        self.epoch_canvas = FigureCanvasQTAgg(self.epoch_figure)
        epoch_layout.addWidget(self.epoch_canvas)

        self.tab_widget.addTab(epoch_tab, "📊 Epoch Metrics")

    def create_batch_tab(self):
        """Crea il tab per le metriche per batch"""
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        # Header con selettore epoca
        header_layout = QHBoxLayout()

        title_label = QLabel("📈 Batch-Level Metrics")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        selector_label = QLabel("Select Epoch:")
        selector_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        header_layout.addWidget(selector_label)

        self.epoch_selector = QComboBox()
        self.epoch_selector.currentIndexChanged.connect(self.on_epoch_selected)
        self.epoch_selector.setStyleSheet("font-size: 11pt; padding: 5px; min-width: 100px;")
        header_layout.addWidget(self.epoch_selector)

        batch_layout.addLayout(header_layout)

        # Canvas per grafici
        self.batch_figure = Figure(figsize=(14, 10), dpi=100)
        self.batch_canvas = FigureCanvasQTAgg(self.batch_figure)
        batch_layout.addWidget(self.batch_canvas)

        self.tab_widget.addTab(batch_tab, "📈 Batch Metrics")

    def create_results_tab(self):
        """Crea il tab per i risultati finali"""
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Titolo
        title_label = QLabel("ℹ️  Training Summary & Test Results")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        results_layout.addWidget(title_label)

        # Text area per informazioni
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("font-family: 'Courier'; font-size: 11pt; padding: 10px;")
        results_layout.addWidget(self.info_text)

        self.tab_widget.addTab(results_tab, "ℹ️  Results & Info")

    def auto_load_latest(self):
        """Carica automaticamente l'ultimo file CSV trovato"""
        csv_files = sorted(glob.glob('logs/train_metrics_*.csv'), reverse=True)
        if csv_files:
            self.load_csv_file(csv_files[0])

    def load_csv(self):
        """Apre dialog per selezionare un CSV"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select training metrics CSV",
            "logs",
            "CSV files (*.csv);;All files (*.*)"
        )
        if filename:
            self.load_csv_file(filename)

    def load_csv_file(self, filename):
        """Carica il file CSV specificato"""
        try:
            self.df = pd.read_csv(filename)
            self.current_csv = filename

            # Estrai timestamp dal nome file
            basename = os.path.basename(filename)
            timestamp = basename.replace('train_metrics_', '').replace('.csv', '')

            # Cerca il file test_results corrispondente
            test_file = f'logs/test_results_{timestamp}.txt'
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    self.test_results = f.read()
            else:
                self.test_results = "Test results file not found."

            # Aggiorna interfaccia
            self.file_label.setText(f"Loaded: {basename}")
            self.update_epoch_selector()
            self.plot_epoch_metrics()
            self.plot_batch_metrics()
            self.update_info()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{str(e)}")

    def refresh_data(self):
        """Ricarica i dati dal file corrente"""
        if self.current_csv:
            self.load_csv_file(self.current_csv)
        else:
            QMessageBox.information(self, "Info", "No file loaded. Please load a CSV first.")

    def update_epoch_selector(self):
        """Aggiorna il selettore delle epoche"""
        if self.df is not None:
            self.epoch_selector.clear()
            epochs = sorted(self.df['epoch'].unique())
            self.epoch_selector.addItems([str(e) for e in epochs])
            if epochs:
                self.epoch_selector.setCurrentIndex(len(epochs) - 1)

    def on_epoch_selected(self):
        """Callback quando viene selezionata un'epoca"""
        self.plot_batch_metrics()

    def plot_epoch_metrics(self):
        """Plotta metriche aggregate per epoca in griglia 2x2"""
        if self.df is None:
            return

        self.epoch_figure.clear()

        # Aggrega per epoca
        agg_dict = {
            'loss': 'mean',
            'dist_similar_mean': 'mean',
            'dist_dissimilar_mean': 'mean'
        }

        # Aggiungi nuove metriche se presenti nel CSV
        if 'db_index' in self.df.columns:
            agg_dict.update({
                'db_index': 'mean',
                'ch_score': 'mean',
                'map': 'mean',
                'nmi': 'mean'
            })

        epoch_stats = self.df.groupby('epoch').agg(agg_dict).reset_index()

        # Determina il numero di subplot in base alle colonne disponibili
        has_new_metrics = 'db_index' in epoch_stats.columns

        # GRIGLIA 2x2
        # Subplot 1 (top-left): Loss
        ax1 = self.epoch_figure.add_subplot(2, 2, 1)
        ax1.plot(epoch_stats['epoch'], epoch_stats['loss'], 'o-', linewidth=2.5, markersize=5, color='crimson')
        ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Average Loss', fontweight='bold', fontsize=11)
        ax1.set_title('Training Loss per Epoch', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Subplot 2 (top-right): Distanze
        ax2 = self.epoch_figure.add_subplot(2, 2, 2)
        ax2.plot(epoch_stats['epoch'], epoch_stats['dist_similar_mean'],
                'o-', label='Similar pairs (l=0)', linewidth=2.5, markersize=5, color='steelblue')
        ax2.plot(epoch_stats['epoch'], epoch_stats['dist_dissimilar_mean'],
                's-', label='Dissimilar pairs (l=1)', linewidth=2.5, markersize=5, color='darkorange')
        ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Average Distance', fontweight='bold', fontsize=11)
        ax2.set_title('Embedding Distances per Epoch', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        # Subplot 3 e 4: Nuove metriche (se disponibili)
        if has_new_metrics:
            # Subplot 3 (bottom-left): DB Index e CH Score
            ax3 = self.epoch_figure.add_subplot(2, 2, 3)
            ax3_twin = ax3.twinx()

            l1 = ax3.plot(epoch_stats['epoch'], epoch_stats['db_index'],
                    'o-', label='DB Index (↓ better)', linewidth=2.5, markersize=5, color='purple')
            ax3.set_xlabel('Epoch', fontweight='bold', fontsize=11)
            ax3.set_ylabel('Davies-Bouldin Index', fontweight='bold', fontsize=10, color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')

            l2 = ax3_twin.plot(epoch_stats['epoch'], epoch_stats['ch_score'],
                    's-', label='CH Score (↑ better)', linewidth=2.5, markersize=5, color='green')
            ax3_twin.set_ylabel('Calinski-Harabasz Score', fontweight='bold', fontsize=10, color='green')
            ax3_twin.tick_params(axis='y', labelcolor='green')

            ax3.set_title('Clustering Quality Metrics', fontweight='bold', fontsize=12)
            ax3.grid(True, alpha=0.3)

            # Combina le leggende
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='best', fontsize=10)

            # Subplot 4 (bottom-right): mAP e NMI
            ax4 = self.epoch_figure.add_subplot(2, 2, 4)
            ax4.plot(epoch_stats['epoch'], epoch_stats['map'],
                    'o-', label='mAP (Mean Avg Precision)', linewidth=2.5, markersize=5, color='teal')
            ax4.plot(epoch_stats['epoch'], epoch_stats['nmi'],
                    's-', label='NMI (Normalized Mutual Info)', linewidth=2.5, markersize=5, color='coral')
            ax4.set_xlabel('Epoch', fontweight='bold', fontsize=11)
            ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
            ax4.set_title('Retrieval & Clustering Alignment', fontweight='bold', fontsize=12)
            ax4.legend(fontsize=10, loc='best')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1.1])  # mAP e NMI sono tra 0 e 1
        else:
            # Se non ci sono nuove metriche, lascia i subplot 3 e 4 vuoti
            ax3 = self.epoch_figure.add_subplot(2, 2, 3)
            ax3.text(0.5, 0.5, 'No additional metrics available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax3.axis('off')

            ax4 = self.epoch_figure.add_subplot(2, 2, 4)
            ax4.axis('off')

        self.epoch_figure.tight_layout()
        self.epoch_canvas.draw()

    def plot_batch_metrics(self):
        """Plotta metriche per batch dell'epoca selezionata in griglia 2x2"""
        if self.df is None or self.epoch_selector.currentText() == '':
            return

        selected_epoch = int(self.epoch_selector.currentText())
        epoch_data = self.df[self.df['epoch'] == selected_epoch]

        self.batch_figure.clear()

        # Determina se ci sono le nuove metriche
        has_new_metrics = 'db_index' in epoch_data.columns

        # GRIGLIA 2x2
        # Subplot 1 (top-left): Loss per batch
        ax1 = self.batch_figure.add_subplot(2, 2, 1)
        ax1.plot(epoch_data['batch'], epoch_data['loss'], 'o-', linewidth=2, markersize=4, color='crimson')
        ax1.set_xlabel('Batch', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax1.set_title(f'Epoch {selected_epoch} - Loss per Batch', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Subplot 2 (top-right): Distanze simili E dissimili (combinato)
        ax2 = self.batch_figure.add_subplot(2, 2, 2)

        # Coppie simili (L=0) in blu
        ax2.plot(epoch_data['batch'], epoch_data['dist_similar_mean'],
                label='Similar (l=0) - Mean', linewidth=2.5, color='steelblue')
        ax2.fill_between(epoch_data['batch'],
                         epoch_data['dist_similar_min'],
                         epoch_data['dist_similar_max'],
                         alpha=0.3, color='steelblue', label='Similar (l=0) - Range')

        # Coppie dissimili (L=1) in arancione
        ax2.plot(epoch_data['batch'], epoch_data['dist_dissimilar_mean'],
                label='Dissimilar (l=1) - Mean', linewidth=2.5, color='darkorange')
        ax2.fill_between(epoch_data['batch'],
                         epoch_data['dist_dissimilar_min'],
                         epoch_data['dist_dissimilar_max'],
                         alpha=0.3, color='darkorange', label='Dissimilar (l=1) - Range')

        ax2.set_xlabel('Batch', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Distance', fontweight='bold', fontsize=11)
        ax2.set_title('Embedding Distances per Batch', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)

        # Subplot 3 e 4: Nuove metriche per batch (se disponibili)
        if has_new_metrics:
            # Subplot 3 (bottom-left): DB Index e CH Score per batch
            ax3 = self.batch_figure.add_subplot(2, 2, 3)
            ax3_twin = ax3.twinx()

            l1 = ax3.plot(epoch_data['batch'], epoch_data['db_index'],
                    'o-', label='DB Index (↓ better)', linewidth=2, markersize=4, color='purple')
            ax3.set_xlabel('Batch', fontweight='bold', fontsize=11)
            ax3.set_ylabel('DB Index', fontweight='bold', fontsize=10, color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')

            l2 = ax3_twin.plot(epoch_data['batch'], epoch_data['ch_score'],
                    's-', label='CH Score (↑ better)', linewidth=2, markersize=4, color='green')
            ax3_twin.set_ylabel('CH Score', fontweight='bold', fontsize=10, color='green')
            ax3_twin.tick_params(axis='y', labelcolor='green')

            ax3.set_title('Clustering Quality per Batch', fontweight='bold', fontsize=12)
            ax3.grid(True, alpha=0.3)

            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='best', fontsize=9)

            # Subplot 4 (bottom-right): mAP e NMI per batch
            ax4 = self.batch_figure.add_subplot(2, 2, 4)
            ax4.plot(epoch_data['batch'], epoch_data['map'],
                    'o-', label='mAP', linewidth=2, markersize=4, color='teal')
            ax4.plot(epoch_data['batch'], epoch_data['nmi'],
                    's-', label='NMI', linewidth=2, markersize=4, color='coral')
            ax4.set_xlabel('Batch', fontweight='bold', fontsize=11)
            ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
            ax4.set_title('Retrieval & Clustering per Batch', fontweight='bold', fontsize=12)
            ax4.legend(fontsize=9, loc='best')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1.1])
        else:
            # Se non ci sono nuove metriche, lascia i subplot 3 e 4 vuoti
            ax3 = self.batch_figure.add_subplot(2, 2, 3)
            ax3.text(0.5, 0.5, 'No additional metrics available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax3.axis('off')

            ax4 = self.batch_figure.add_subplot(2, 2, 4)
            ax4.axis('off')

        self.batch_figure.tight_layout()
        self.batch_canvas.draw()

    def update_info(self):
        """Aggiorna il box informazioni"""
        if self.df is None:
            return

        # Statistiche generali
        total_epochs = self.df['epoch'].max()
        total_batches = self.df['batch'].max()
        total_samples = len(self.df)

        info = f"{'='*70}\n"
        info += f"TRAINING SUMMARY\n"
        info += f"{'='*70}\n"
        info += f"CSV File: {os.path.basename(self.current_csv)}\n"
        info += f"Total Epochs: {total_epochs}\n"
        info += f"Batches per Epoch: {total_batches}\n"
        info += f"Total Samples: {total_samples}\n"
        info += f"\n--- Final Metrics (Last Batch) ---\n"
        info += f"Loss: {self.df['loss'].iloc[-1]:.4f}\n"
        info += f"Distance (similar pairs): {self.df['dist_similar_mean'].iloc[-1]:.4f}\n"
        info += f"Distance (dissimilar pairs): {self.df['dist_dissimilar_mean'].iloc[-1]:.4f}\n"

        # Aggiungi nuove metriche se presenti
        if 'db_index' in self.df.columns:
            info += f"\n--- Embedding Quality Metrics (Last Batch) ---\n"
            info += f"Davies-Bouldin Index: {self.df['db_index'].iloc[-1]:.4f} (lower is better)\n"
            info += f"Calinski-Harabasz Score: {self.df['ch_score'].iloc[-1]:.2f} (higher is better)\n"
            info += f"Mean Average Precision (mAP): {self.df['map'].iloc[-1]:.4f}\n"
            info += f"Normalized Mutual Information (NMI): {self.df['nmi'].iloc[-1]:.4f}\n"

        info += f"\n{'='*70}\n"
        info += f"{self.test_results}\n"

        self.info_text.setText(info)

    def export_plots(self):
        """Esporta i grafici come immagini PNG"""
        if self.df is None:
            QMessageBox.information(self, "Info", "No data loaded. Please load a CSV first.")
            return

        try:
            timestamp = os.path.basename(self.current_csv).replace('train_metrics_', '').replace('.csv', '')

            # Salva grafico epoche
            self.epoch_figure.savefig(f'logs/plot_epochs_{timestamp}.png', dpi=150, bbox_inches='tight')

            # Salva grafico batch
            self.batch_figure.savefig(f'logs/plot_batch_{timestamp}.png', dpi=150, bbox_inches='tight')

            QMessageBox.information(self, "Success",
                                   f"Plots exported to logs/ directory:\n"
                                   f"- plot_epochs_{timestamp}.png\n"
                                   f"- plot_batch_{timestamp}.png")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export plots:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    viewer = MetricViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
