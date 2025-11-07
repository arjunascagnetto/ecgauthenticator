#!/usr/bin/env python3
"""
Script per lanciare la UI di configurazione del training (PyQt5)
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from src.train_config_ui import TrainingConfigUI


def main():
    """Main function to launch the training config UI"""
    app = QApplication(sys.argv)

    config_path = str(Path(__file__).parent / 'train_configs.yaml')
    ui = TrainingConfigUI(config_path)
    ui.show()

    print("\n" + "="*70)
    print("ECG Metric Learning - Training Configuration UI")
    print("="*70)
    print(f"Config file: {config_path}")
    print("\nConfigure parameters and click 'Save Config'")
    print("Then run: python src/train_metric_ecg.py")
    print("="*70 + "\n")

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
