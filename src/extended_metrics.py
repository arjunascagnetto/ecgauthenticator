"""Metriche estese per clustering evaluation."""
import numpy as np
from typing import Dict, List
from sklearn.metrics import silhouette_score


def compute_extended_metrics(embeddings: np.ndarray, patient_ids: np.ndarray) -> Dict:
    """
    Calcola metriche estese veloci: intra/inter ratio, compactness, separation.

    Args:
        embeddings: (N, D) array
        patient_ids: (N,) array di patient IDs

    Returns:
        dict con metriche aggiuntive
    """
    metrics = {}
    unique_patients = np.unique(patient_ids)

    # 1. Intra/Inter distances ratio (veloce con sampling)
    intra_dists = []
    inter_dists = []

    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_embs = embeddings[mask]
        if len(patient_embs) > 1:
            dists = np.linalg.norm(patient_embs[:, np.newaxis] - patient_embs[np.newaxis, :], axis=2)
            upper_idx = np.triu_indices(len(patient_embs), k=1)
            intra_dists.extend(dists[upper_idx])

    # Sample inter-distances se troppi pazienti
    unique_pairs = list(enumerate(unique_patients))
    for i in range(min(len(unique_pairs), 50)):
        for j in range(i + 1, min(len(unique_pairs), 50)):
            p1 = unique_patients[i]
            p2 = unique_patients[j]
            mask1 = patient_ids == p1
            mask2 = patient_ids == p2
            dists = np.linalg.norm(embeddings[mask1][:, np.newaxis] - embeddings[mask2][np.newaxis, :], axis=2)
            inter_dists.extend(dists.flatten()[:100])  # sample max 100 distanze

    intra_mean = np.mean(intra_dists) if intra_dists else 1.0
    inter_mean = np.mean(inter_dists) if inter_dists else 1.0
    metrics['intra_inter_ratio'] = float(inter_mean / intra_mean) if intra_mean > 0 else 0.0

    # 2. Cluster compactness: max intra-cluster distance per paziente
    max_intra_dists = []
    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_embs = embeddings[mask]
        if len(patient_embs) > 1:
            dists = np.linalg.norm(patient_embs[:, np.newaxis] - patient_embs[np.newaxis, :], axis=2)
            max_intra_dists.append(np.max(dists))
    metrics['cluster_compactness'] = float(np.mean(max_intra_dists)) if max_intra_dists else 0.0

    # 3. Inter-cluster separation: min inter-cluster distance (sampled)
    min_inter_dists = []
    for i in range(min(len(unique_patients), 50)):
        for j in range(i + 1, min(len(unique_patients), 50)):
            p1 = unique_patients[i]
            p2 = unique_patients[j]
            mask1 = patient_ids == p1
            mask2 = patient_ids == p2
            dists = np.linalg.norm(embeddings[mask1][:, np.newaxis] - embeddings[mask2][np.newaxis, :], axis=2)
            min_inter_dists.append(np.min(dists))
    metrics['cluster_separation'] = float(np.mean(min_inter_dists)) if min_inter_dists else 0.0

    return metrics


def compute_extended_metrics_selective(embeddings: np.ndarray, patient_ids: np.ndarray,
                                       metrics_to_compute: List[str]) -> Dict:
    """
    Calcola solo metriche selezionate.

    Args:
        embeddings: (N, D) array
        patient_ids: (N,) array di patient IDs
        metrics_to_compute: lista di nomi metriche da calcolare

    Returns:
        dict con metriche richieste
    """
    metrics = {}
    unique_patients = np.unique(patient_ids)

    if 'silhouette' in metrics_to_compute:
        try:
            if len(embeddings) > 5000:
                idx = np.random.choice(len(embeddings), 5000, replace=False)
                sil = silhouette_score(embeddings[idx], patient_ids[idx], sample_size=1000)
            else:
                sil = silhouette_score(embeddings, patient_ids, sample_size=1000)
            metrics['silhouette'] = float(sil)
        except:
            metrics['silhouette'] = -1.0

    if 'nn_accuracy' in metrics_to_compute:
        if len(embeddings) > 1:
            dists = np.linalg.norm(embeddings[:, np.newaxis] - embeddings[np.newaxis, :], axis=2)
            nn_idx = np.argsort(dists, axis=1)[:, 1]
            nn_accuracy = np.mean(patient_ids[nn_idx] == patient_ids) * 100
            metrics['nn_accuracy'] = float(nn_accuracy)
        else:
            metrics['nn_accuracy'] = 0.0

    if 'intra_inter_ratio' in metrics_to_compute:
        intra_dists = []
        inter_dists = []
        for patient_id in unique_patients:
            mask = patient_ids == patient_id
            patient_embs = embeddings[mask]
            if len(patient_embs) > 1:
                dists = np.linalg.norm(patient_embs[:, np.newaxis] - patient_embs[np.newaxis, :], axis=2)
                upper_idx = np.triu_indices(len(patient_embs), k=1)
                intra_dists.extend(dists[upper_idx])
        for i in range(min(len(unique_patients), 50)):
            for j in range(i + 1, min(len(unique_patients), 50)):
                p1 = unique_patients[i]
                p2 = unique_patients[j]
                mask1 = patient_ids == p1
                mask2 = patient_ids == p2
                dists = np.linalg.norm(embeddings[mask1][:, np.newaxis] - embeddings[mask2][np.newaxis, :], axis=2)
                inter_dists.extend(dists.flatten()[:100])
        intra_mean = np.mean(intra_dists) if intra_dists else 1.0
        inter_mean = np.mean(inter_dists) if inter_dists else 1.0
        metrics['intra_inter_ratio'] = float(inter_mean / intra_mean) if intra_mean > 0 else 0.0

    if 'cluster_compactness' in metrics_to_compute:
        max_intra_dists = []
        for patient_id in unique_patients:
            mask = patient_ids == patient_id
            patient_embs = embeddings[mask]
            if len(patient_embs) > 1:
                dists = np.linalg.norm(patient_embs[:, np.newaxis] - patient_embs[np.newaxis, :], axis=2)
                max_intra_dists.append(np.max(dists))
        metrics['cluster_compactness'] = float(np.mean(max_intra_dists)) if max_intra_dists else 0.0

    if 'cluster_separation' in metrics_to_compute:
        min_inter_dists = []
        for i in range(min(len(unique_patients), 50)):
            for j in range(i + 1, min(len(unique_patients), 50)):
                p1 = unique_patients[i]
                p2 = unique_patients[j]
                mask1 = patient_ids == p1
                mask2 = patient_ids == p2
                dists = np.linalg.norm(embeddings[mask1][:, np.newaxis] - embeddings[mask2][np.newaxis, :], axis=2)
                min_inter_dists.append(np.min(dists))
        metrics['cluster_separation'] = float(np.mean(min_inter_dists)) if min_inter_dists else 0.0

    return metrics
