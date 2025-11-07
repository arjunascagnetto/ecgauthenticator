import yaml
import json
from pathlib import Path
from typing import Dict, Any
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QCheckBox, QPushButton,
                             QComboBox, QScrollArea, QMessageBox, QFileDialog,
                             QGroupBox, QApplication, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QTabWidget)
from PyQt5.QtCore import Qt


class TrainingConfigUI(QMainWindow):
    """UI PyQt5 per configurare i parametri di training"""

    DEFAULT_CONFIG = {
        'training': {
            'use_curriculum': False,
            'max_epochs': 50,
            'patience': 15,
            'early_stop_ch_threshold': 5.0,
        },
        'optimizer': {
            'learning_rate': 5.0e-4,
            'weight_decay': 1.0e-5,
            'lr_scheduler_factor': 0.5,
            'lr_scheduler_patience': 5,
        },
        'loss': {
            'loss_type': 'curriculum_contrastive',
            'margin_init': 1.0,
            'margin_final': 0.5,
            'alpha': 2.0,
            'beta': 50.0,
            'lambda_param': 0.5,
        },
        'mining': {
            'strategy': 'hard',
            'start_epoch': 1,
            'random_epochs': 10,
            'semihard_epochs': 25,
            'hardmining_epochs': 50,
        },
        'batch': {
            'use_pk_sampling': True,
            'batch_size': 256,
            'num_patients_per_batch': 64,
            'num_ecg_per_patient': 4,
            'shuffle': True,
        },
        'encoder': {
            'input_dim': 13,
            'hidden_dim': 20,
            'embedding_dim': 32,
            'dropout': 0.2,
            'normalize': True,
        },
        'validation': {
            'sample_size': 5000,
            'compute_silhouette': False,
            'compute_bw_ratio': False,
        },
        'test': {
            'compute_bw_ratio': True,
        },
        'device': 'cpu',
    }

    def __init__(self, config_path: str = 'train_configs.yaml'):
        super().__init__()
        self.config_path = config_path
        self.config = self.load_config()
        self.entries = {}

        self.setWindowTitle("ECG Metric Learning - Training Configuration")
        self.setGeometry(100, 100, 1200, 700)

        self.create_ui()

    def load_config(self) -> Dict[str, Any]:
        """Carica config da YAML o usa defaults"""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def create_ui(self):
        """Crea interfaccia grafica con PyQt5 usando Tabs"""
        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principale
        main_layout = QVBoxLayout(central_widget)

        # Crea tab widget
        tabs = QTabWidget()

        # Aggiungi tab per ogni sezione
        tabs.addTab(self.create_section("TRAINING", self.config['training']), "Training")
        tabs.addTab(self.create_section("OPTIMIZER", self.config['optimizer']), "Optimizer")
        tabs.addTab(self.create_loss_tab(), "Loss")
        tabs.addTab(self.create_mining_tab(), "Mining")
        tabs.addTab(self.create_batch_tab(), "Batch")
        tabs.addTab(self.create_section("ENCODER", self.config['encoder']), "Encoder")
        tabs.addTab(self.create_validation_device_tab(), "Validation & Device")

        main_layout.addWidget(tabs)

        # Button layout
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self.load_config_dialog)
        button_layout.addWidget(load_btn)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        button_layout.addWidget(reset_btn)

        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)

        main_layout.addLayout(button_layout)

    def create_section(self, title: str, section_config: Dict, exclude_keys=None) -> QGroupBox:
        """Crea una sezione di configurazione"""
        if exclude_keys is None:
            exclude_keys = []

        group = QGroupBox(title)
        layout = QFormLayout()

        for key, value in section_config.items():
            if key in exclude_keys:
                continue

            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                self.entries[f"{title}_{key}"] = widget
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setValue(value)
                widget.setRange(-9999, 9999)
                self.entries[f"{title}_{key}"] = widget
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setValue(value)
                widget.setDecimals(10)
                widget.setRange(-9999.0, 9999.0)
                self.entries[f"{title}_{key}"] = widget
            elif isinstance(value, str):
                widget = QLineEdit()
                widget.setText(value)
                self.entries[f"{title}_{key}"] = widget
            else:
                widget = QLineEdit()
                widget.setText(str(value))
                self.entries[f"{title}_{key}"] = widget

            layout.addRow(f"{key}:", widget)

        group.setLayout(layout)
        return group

    def create_loss_tab(self) -> QWidget:
        """Crea tab per Loss con loss_type dropdown"""
        container = QWidget()
        layout = QVBoxLayout(container)

        group = QGroupBox("LOSS")
        form_layout = QFormLayout()

        # Loss type dropdown
        loss_combo = QComboBox()
        loss_types = ['contrastive', 'adaptive_contrastive', 'curriculum_contrastive', 'multi_similarity']
        loss_combo.addItems(loss_types)
        loss_combo.setCurrentText(self.config['loss']['loss_type'])
        self.entries['LOSS_loss_type'] = loss_combo
        form_layout.addRow("loss_type:", loss_combo)

        # Margin init
        margin_init_widget = QDoubleSpinBox()
        margin_init_widget.setValue(self.config['loss']['margin_init'])
        margin_init_widget.setDecimals(10)
        margin_init_widget.setRange(-9999.0, 9999.0)
        self.entries['LOSS_margin_init'] = margin_init_widget
        form_layout.addRow("margin_init:", margin_init_widget)

        # Margin final
        margin_final_widget = QDoubleSpinBox()
        margin_final_widget.setValue(self.config['loss']['margin_final'])
        margin_final_widget.setDecimals(10)
        margin_final_widget.setRange(-9999.0, 9999.0)
        self.entries['LOSS_margin_final'] = margin_final_widget
        form_layout.addRow("margin_final:", margin_final_widget)

        # Alpha (for multi_similarity)
        alpha_widget = QDoubleSpinBox()
        alpha_widget.setValue(self.config['loss']['alpha'])
        alpha_widget.setDecimals(10)
        alpha_widget.setRange(-9999.0, 9999.0)
        self.entries['LOSS_alpha'] = alpha_widget
        form_layout.addRow("alpha:", alpha_widget)

        # Beta (for multi_similarity)
        beta_widget = QDoubleSpinBox()
        beta_widget.setValue(self.config['loss']['beta'])
        beta_widget.setDecimals(10)
        beta_widget.setRange(-9999.0, 9999.0)
        self.entries['LOSS_beta'] = beta_widget
        form_layout.addRow("beta:", beta_widget)

        # Lambda param (for multi_similarity)
        lambda_widget = QDoubleSpinBox()
        lambda_widget.setValue(self.config['loss']['lambda_param'])
        lambda_widget.setDecimals(10)
        lambda_widget.setRange(-9999.0, 9999.0)
        self.entries['LOSS_lambda_param'] = lambda_widget
        form_layout.addRow("lambda_param:", lambda_widget)

        group.setLayout(form_layout)
        layout.addWidget(group)
        layout.addStretch()

        return container

    def create_mining_tab(self) -> QWidget:
        """Crea tab per Mining con strategy dropdown"""
        container = QWidget()
        layout = QVBoxLayout(container)

        group = QGroupBox("MINING")
        form_layout = QFormLayout()

        # Mining strategy dropdown
        strategy_combo = QComboBox()
        strategies = ['random', 'semi-hard', 'hard']
        strategy_combo.addItems(strategies)
        strategy_combo.setCurrentText(self.config['mining'].get('strategy', 'hard'))
        self.entries['MINING_strategy'] = strategy_combo
        form_layout.addRow("strategy:", strategy_combo)

        # Start epoch
        start_epoch_widget = QSpinBox()
        start_epoch_widget.setValue(self.config['mining'].get('start_epoch', 1))
        start_epoch_widget.setRange(1, 9999)
        self.entries['MINING_start_epoch'] = start_epoch_widget
        form_layout.addRow("start_epoch:", start_epoch_widget)

        # Legacy parameters (per retrocompatibilità curriculum)
        random_epochs_widget = QSpinBox()
        random_epochs_widget.setValue(self.config['mining']['random_epochs'])
        random_epochs_widget.setRange(1, 9999)
        self.entries['MINING_random_epochs'] = random_epochs_widget
        form_layout.addRow("random_epochs:", random_epochs_widget)

        semihard_epochs_widget = QSpinBox()
        semihard_epochs_widget.setValue(self.config['mining']['semihard_epochs'])
        semihard_epochs_widget.setRange(1, 9999)
        self.entries['MINING_semihard_epochs'] = semihard_epochs_widget
        form_layout.addRow("semihard_epochs:", semihard_epochs_widget)

        hardmining_epochs_widget = QSpinBox()
        hardmining_epochs_widget.setValue(self.config['mining']['hardmining_epochs'])
        hardmining_epochs_widget.setRange(1, 9999)
        self.entries['MINING_hardmining_epochs'] = hardmining_epochs_widget
        form_layout.addRow("hardmining_epochs:", hardmining_epochs_widget)

        group.setLayout(form_layout)
        layout.addWidget(group)
        layout.addStretch()

        return container

    def create_batch_tab(self) -> QWidget:
        """Crea tab per Batch con logica PK sampling"""
        container = QWidget()
        layout = QVBoxLayout(container)

        group = QGroupBox("BATCH")
        form_layout = QFormLayout()

        # Use PK sampling checkbox
        pk_checkbox = QCheckBox()
        pk_checkbox.setChecked(self.config['batch']['use_pk_sampling'])
        self.entries['BATCH_use_pk_sampling'] = pk_checkbox
        form_layout.addRow("use_pk_sampling:", pk_checkbox)

        # Num patients per batch
        num_patients_widget = QSpinBox()
        num_patients_widget.setValue(self.config['batch']['num_patients_per_batch'])
        num_patients_widget.setRange(1, 9999)
        self.entries['BATCH_num_patients_per_batch'] = num_patients_widget
        form_layout.addRow("num_patients_per_batch (P):", num_patients_widget)

        # Num ECG per patient
        num_ecg_widget = QSpinBox()
        num_ecg_widget.setValue(self.config['batch']['num_ecg_per_patient'])
        num_ecg_widget.setRange(1, 9999)
        self.entries['BATCH_num_ecg_per_patient'] = num_ecg_widget
        form_layout.addRow("num_ecg_per_patient (K):", num_ecg_widget)

        # Batch size (disabled if PK is enabled)
        batch_size_widget = QSpinBox()
        batch_size_widget.setValue(self.config['batch']['batch_size'])
        batch_size_widget.setRange(1, 9999)
        batch_size_widget.setEnabled(not self.config['batch']['use_pk_sampling'])
        self.entries['BATCH_batch_size'] = batch_size_widget
        form_layout.addRow("batch_size:", batch_size_widget)

        # Shuffle
        shuffle_widget = QCheckBox()
        shuffle_widget.setChecked(self.config['batch']['shuffle'])
        self.entries['BATCH_shuffle'] = shuffle_widget
        form_layout.addRow("shuffle:", shuffle_widget)

        # Connect signals to update batch_size when P or K change (if PK enabled)
        def update_batch_size():
            if pk_checkbox.isChecked():
                p = num_patients_widget.value()
                k = num_ecg_widget.value()
                batch_size_widget.setValue(p * k)
                batch_size_widget.setEnabled(False)
            else:
                batch_size_widget.setEnabled(True)

        pk_checkbox.stateChanged.connect(update_batch_size)
        num_patients_widget.valueChanged.connect(update_batch_size)
        num_ecg_widget.valueChanged.connect(update_batch_size)

        # Initial update
        update_batch_size()

        group.setLayout(form_layout)
        layout.addWidget(group)
        layout.addStretch()

        return container

    def create_validation_device_tab(self) -> QWidget:
        """Crea tab per Validation e Device"""
        container = QWidget()
        layout = QVBoxLayout(container)

        # Validation section
        validation_group = QGroupBox("VALIDATION")
        validation_layout = QFormLayout()

        sample_size_widget = QSpinBox()
        sample_size_widget.setValue(self.config['validation']['sample_size'])
        sample_size_widget.setRange(1, 99999)
        self.entries['VALIDATION_sample_size'] = sample_size_widget
        validation_layout.addRow("sample_size:", sample_size_widget)

        silhouette_widget = QCheckBox()
        silhouette_widget.setChecked(self.config['validation']['compute_silhouette'])
        self.entries['VALIDATION_compute_silhouette'] = silhouette_widget
        validation_layout.addRow("compute_silhouette:", silhouette_widget)

        bw_ratio_widget = QCheckBox()
        bw_ratio_widget.setChecked(self.config['validation']['compute_bw_ratio'])
        self.entries['VALIDATION_compute_bw_ratio'] = bw_ratio_widget
        validation_layout.addRow("compute_bw_ratio:", bw_ratio_widget)

        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)

        # Device section
        device_group = QGroupBox("DEVICE")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        device_combo = QComboBox()
        device_combo.addItems(['cpu', 'cuda'])
        device_combo.setCurrentText(self.config.get('device', 'cpu'))
        self.entries['device'] = device_combo
        device_layout.addWidget(device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        layout.addStretch()

        return container

    def save_config(self):
        """Salva configurazione in YAML"""
        try:
            # Ricostruisci config da entries
            new_config = self.DEFAULT_CONFIG.copy()

            # Aggiorna training
            new_config['training']['use_curriculum'] = self.entries['TRAINING_use_curriculum'].isChecked()
            new_config['training']['max_epochs'] = int(self.entries['TRAINING_max_epochs'].value())
            new_config['training']['patience'] = int(self.entries['TRAINING_patience'].value())
            new_config['training']['early_stop_ch_threshold'] = float(
                self.entries['TRAINING_early_stop_ch_threshold'].value())

            # Aggiorna optimizer
            new_config['optimizer']['learning_rate'] = float(self.entries['OPTIMIZER_learning_rate'].value())
            new_config['optimizer']['weight_decay'] = float(self.entries['OPTIMIZER_weight_decay'].value())
            new_config['optimizer']['lr_scheduler_factor'] = float(
                self.entries['OPTIMIZER_lr_scheduler_factor'].value())
            new_config['optimizer']['lr_scheduler_patience'] = int(
                self.entries['OPTIMIZER_lr_scheduler_patience'].value())

            # Aggiorna loss
            new_config['loss']['loss_type'] = self.entries['LOSS_loss_type'].currentText()
            new_config['loss']['margin_init'] = float(self.entries['LOSS_margin_init'].value())
            new_config['loss']['margin_final'] = float(self.entries['LOSS_margin_final'].value())
            new_config['loss']['alpha'] = float(self.entries['LOSS_alpha'].value())
            new_config['loss']['beta'] = float(self.entries['LOSS_beta'].value())
            new_config['loss']['lambda_param'] = float(self.entries['LOSS_lambda_param'].value())

            # Aggiorna mining
            new_config['mining']['strategy'] = self.entries['MINING_strategy'].currentText()
            new_config['mining']['start_epoch'] = int(self.entries['MINING_start_epoch'].value())
            new_config['mining']['random_epochs'] = int(self.entries['MINING_random_epochs'].value())
            new_config['mining']['semihard_epochs'] = int(self.entries['MINING_semihard_epochs'].value())
            new_config['mining']['hardmining_epochs'] = int(self.entries['MINING_hardmining_epochs'].value())

            # Aggiorna batch
            new_config['batch']['use_pk_sampling'] = self.entries['BATCH_use_pk_sampling'].isChecked()
            new_config['batch']['batch_size'] = int(self.entries['BATCH_batch_size'].value())
            new_config['batch']['num_patients_per_batch'] = int(
                self.entries['BATCH_num_patients_per_batch'].value())
            new_config['batch']['num_ecg_per_patient'] = int(
                self.entries['BATCH_num_ecg_per_patient'].value())
            new_config['batch']['shuffle'] = self.entries['BATCH_shuffle'].isChecked()

            # Aggiorna encoder
            new_config['encoder']['input_dim'] = int(self.entries['ENCODER_input_dim'].value())
            new_config['encoder']['hidden_dim'] = int(self.entries['ENCODER_hidden_dim'].value())
            new_config['encoder']['embedding_dim'] = int(self.entries['ENCODER_embedding_dim'].value())
            new_config['encoder']['dropout'] = float(self.entries['ENCODER_dropout'].value())
            new_config['encoder']['normalize'] = self.entries['ENCODER_normalize'].isChecked()

            # Aggiorna validation
            new_config['validation']['sample_size'] = int(self.entries['VALIDATION_sample_size'].value())
            new_config['validation']['compute_silhouette'] = self.entries['VALIDATION_compute_silhouette'].isChecked()
            new_config['validation']['compute_bw_ratio'] = self.entries['VALIDATION_compute_bw_ratio'].isChecked()

            # Aggiorna device
            new_config['device'] = self.entries['device'].currentText()

            # Assicurati che test sezione esiste
            if 'test' not in new_config:
                new_config['test'] = {'compute_bw_ratio': True}

            # Salva in YAML
            with open(self.config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

            QMessageBox.information(self, "Success", f"Configuration saved to {self.config_path}")
            self.config = new_config

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving config: {str(e)}")

    def load_config_dialog(self):
        """Dialog per caricare config da file"""
        file, _ = QFileDialog.getOpenFileName(self, "Load Config", "",
                                              "YAML files (*.yaml);;All files (*)")
        if file:
            try:
                with open(file, 'r') as f:
                    self.config = yaml.safe_load(f)
                QMessageBox.information(self, "Success", f"Configuration loaded from {file}")
                self.refresh_ui()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading config: {str(e)}")

    def reset_defaults(self):
        """Ripristina valori di default"""
        if QMessageBox.question(self, "Confirm", "Reset to default values?",
                               QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.config = self.DEFAULT_CONFIG.copy()
            self.refresh_ui()

    def refresh_ui(self):
        """Aggiorna UI con nuovi valori config"""
        # Training
        self.entries['TRAINING_use_curriculum'].setChecked(self.config['training']['use_curriculum'])
        self.entries['TRAINING_max_epochs'].setValue(self.config['training']['max_epochs'])
        self.entries['TRAINING_patience'].setValue(self.config['training']['patience'])
        self.entries['TRAINING_early_stop_ch_threshold'].setValue(
            self.config['training']['early_stop_ch_threshold'])

        # Optimizer
        self.entries['OPTIMIZER_learning_rate'].setValue(self.config['optimizer']['learning_rate'])
        self.entries['OPTIMIZER_weight_decay'].setValue(self.config['optimizer']['weight_decay'])
        self.entries['OPTIMIZER_lr_scheduler_factor'].setValue(
            self.config['optimizer']['lr_scheduler_factor'])
        self.entries['OPTIMIZER_lr_scheduler_patience'].setValue(
            self.config['optimizer']['lr_scheduler_patience'])

        # Loss
        self.entries['LOSS_loss_type'].setCurrentText(self.config['loss']['loss_type'])
        self.entries['LOSS_margin_init'].setValue(self.config['loss']['margin_init'])
        self.entries['LOSS_margin_final'].setValue(self.config['loss']['margin_final'])
        self.entries['LOSS_alpha'].setValue(self.config['loss']['alpha'])
        self.entries['LOSS_beta'].setValue(self.config['loss']['beta'])
        self.entries['LOSS_lambda_param'].setValue(self.config['loss']['lambda_param'])

        # Mining
        self.entries['MINING_strategy'].setCurrentText(self.config['mining'].get('strategy', 'hard'))
        self.entries['MINING_start_epoch'].setValue(self.config['mining'].get('start_epoch', 1))
        self.entries['MINING_random_epochs'].setValue(self.config['mining']['random_epochs'])
        self.entries['MINING_semihard_epochs'].setValue(self.config['mining']['semihard_epochs'])
        self.entries['MINING_hardmining_epochs'].setValue(self.config['mining']['hardmining_epochs'])

        # Batch
        self.entries['BATCH_use_pk_sampling'].setChecked(self.config['batch']['use_pk_sampling'])
        self.entries['BATCH_batch_size'].setValue(self.config['batch']['batch_size'])
        self.entries['BATCH_num_patients_per_batch'].setValue(
            self.config['batch']['num_patients_per_batch'])
        self.entries['BATCH_num_ecg_per_patient'].setValue(
            self.config['batch']['num_ecg_per_patient'])
        self.entries['BATCH_shuffle'].setChecked(self.config['batch']['shuffle'])

        # Encoder
        self.entries['ENCODER_input_dim'].setValue(self.config['encoder']['input_dim'])
        self.entries['ENCODER_hidden_dim'].setValue(self.config['encoder']['hidden_dim'])
        self.entries['ENCODER_embedding_dim'].setValue(self.config['encoder']['embedding_dim'])
        self.entries['ENCODER_dropout'].setValue(self.config['encoder']['dropout'])
        self.entries['ENCODER_normalize'].setChecked(self.config['encoder']['normalize'])

        # Validation
        self.entries['VALIDATION_sample_size'].setValue(self.config['validation']['sample_size'])
        self.entries['VALIDATION_compute_silhouette'].setChecked(
            self.config['validation']['compute_silhouette'])
        self.entries['VALIDATION_compute_bw_ratio'].setChecked(
            self.config['validation']['compute_bw_ratio'])

        # Device
        self.entries['device'].setCurrentText(self.config['device'])
