"""
Grid search sweep script.
Legge exps_config.yaml, genera tutte le combinazioni di parametri,
modifica train_config_v2.yaml e esegue trainer_v2.py per ogni combinazione.
"""

import yaml
import itertools
import subprocess
import sys
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    """Carica file YAML"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    """Salva file YAML"""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def flatten_config(config, parent_key=''):
    """Converte config nested in lista di (key_path, values)"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key))
        elif isinstance(v, list):
            items.append((new_key, v))
        else:
            items.append((new_key, [v]))  # Wrappa valori singoli in lista
    return items


def set_nested_value(d, key_path, value):
    """Setta un valore in dict nested usando path tipo 'optimizer.learning_rate'"""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def generate_combinations(exps_config):
    """Genera tutte le combinazioni di parametri"""
    flat_items = flatten_config(exps_config)
    param_names = [item[0] for item in flat_items]
    param_values = [item[1] for item in flat_items]

    # Genera il cartesian product
    for combo in itertools.product(*param_values):
        yield dict(zip(param_names, combo))


def run_sweep():
    """Esegue il grid search"""
    base_dir = Path(__file__).parent
    exps_config_path = base_dir / 'exps_config.yaml'
    train_config_path = base_dir / 'train_config_v2.yaml'

    # Carica config di sweep e template base
    exps_config = load_yaml(exps_config_path)
    base_config = load_yaml(train_config_path)

    # Genera combinazioni
    combinations = list(generate_combinations(exps_config))

    print(f"\n{'='*80}")
    print(f"🔍 Grid Search: {len(combinations)} combinazioni trovate")
    print(f"{'='*80}\n")

    # Mostra preview primi 5 parametri
    for i, combo in enumerate(combinations[:5]):
        params_str = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(combo.items())
                               if k not in ['data.train_csv', 'data.val_csv', 'data.test_csv', 'device'])
        print(f"  {i+1}. {params_str}")

    if len(combinations) > 5:
        print(f"  ... e {len(combinations) - 5} altre combinazioni\n")
    else:
        print()

    # Chiedi conferma
    while True:
        response = input(f"Vuoi eseguire {len(combinations)} run? (s/n): ").strip().lower()
        if response in ['s', 'si', 'yes', 'y']:
            print("\n✅ Inizio sweep...\n")
            break
        elif response in ['n', 'no']:
            print("\n❌ Sweep annullato.\n")
            sys.exit(0)
        else:
            print("❌ Input non valido. Rispondi 's' oppure 'n'")

    failed_runs = []
    successful_runs = []

    for i, combo in enumerate(combinations, 1):
        # Crea config per questa iterazione
        config = deepcopy(base_config)
        for param_path, value in combo.items():
            set_nested_value(config, param_path, value)

        # Salva config modificato
        save_yaml(config, train_config_path)

        # Genera nome run da parametri
        run_name = "_".join(f"{k.split('.')[-1]}={v}".replace(' ', '').replace('[', '').replace(']', '')
                            for k, v in sorted(combo.items())
                            if k not in ['data.train_csv', 'data.val_csv', 'data.test_csv', 'device'])

        print(f"\n[{i}/{len(combinations)}] Esecuzione: {run_name}")
        print(f"Parametri: {combo}")
        print("-" * 80)

        # Esegui trainer
        try:
            result = subprocess.run(
                [sys.executable, 'trainer_v2.py'],
                cwd=base_dir,
                capture_output=False
            )

            if result.returncode == 0:
                print(f"✅ Run {i} completata con successo")
                successful_runs.append(run_name)
            else:
                print(f"❌ Run {i} fallita con codice {result.returncode}")
                failed_runs.append((run_name, result.returncode))
        except Exception as e:
            print(f"❌ Errore durante esecuzione run {i}: {e}")
            failed_runs.append((run_name, str(e)))

    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Completate: {len(successful_runs)}/{len(combinations)}")
    print(f"❌ Fallite: {len(failed_runs)}/{len(combinations)}")

    if failed_runs:
        print("\nRun fallite:")
        for run_name, error in failed_runs:
            print(f"  - {run_name}: {error}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_sweep()
