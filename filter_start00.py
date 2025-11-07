import pandas as pd

# Carica i dati
print("Caricamento start00.csv...")
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/start00.csv')

print(f"Righe originali: {len(df)}")
print(f"Pazienti unici: {df['PatientID'].nunique()}")

# Conta gli esami per paziente
exams_per_patient = df['PatientID'].value_counts()
print(f"\nDistribuzione esami per paziente:")
print(exams_per_patient.value_counts().sort_index())

# Filtra: tieni solo pazienti con 2+ esami
patients_with_multiple_exams = exams_per_patient[exams_per_patient >= 2].index
df_filtered = df[df['PatientID'].isin(patients_with_multiple_exams)].copy()

print(f"\nDopo filtro:")
print(f"Righe rimaste: {len(df_filtered)}")
print(f"Pazienti rimasti: {df_filtered['PatientID'].nunique()}")
print(f"Righe eliminate: {len(df) - len(df_filtered)}")

# Sovrascrivi il file
df_filtered.to_csv('/Users/arjuna/Progetti/siamese/data/start00.csv', index=False)
print(f"\nFile sovrascritto: /Users/arjuna/Progetti/siamese/data/start00.csv")

