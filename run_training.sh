#!/bin/bash

# Script per lanciare il training ECG Metric Learning
# Usa la configurazione in train_configs.yaml

set -e

VENV="/Users/arjuna/Progetti/siamese/.siamese"
PYTHON="$VENV/bin/python"

echo "=================================================="
echo "ECG Metric Learning Training"
echo "=================================================="
echo ""
echo "Configuration file: train_configs.yaml"
echo ""

# Verifica che il venv esiste
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV"
    exit 1
fi

# Esegui training
cd /Users/arjuna/Progetti/siamese
$PYTHON src/train_metric_ecg.py

echo ""
echo "Training completed!"
echo "Check logs/ and results/ directories for output"
