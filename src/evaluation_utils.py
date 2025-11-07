import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import logging
import torch

logger = logging.getLogger(__name__)


def compute_batch_metrics(embeddings: np.ndarray, patient_ids: np.ndarray) -> Dict:
    """
    Calcola metriche per un batch di embeddings.

    Args:
        embeddings: (N, D) array di embeddings
        patient_ids: (N,) array di patient IDs

    Returns:
        dict con:
        - d_intra: media distanze cosine intra-paziente (same patient)
        - d_inter: media distanze cosine inter-paziente (different patient)
        - db: Davies-Bouldin index
        - ch: Calinski-Harabasz index
    """
    metrics = {}

    # Calcola distanze intra e inter paziente
    unique_patients = np.unique(patient_ids)
    intra_distances = []
    inter_distances = []

    # Intra-patient distances
    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_embs = embeddings[mask]

        if len(patient_embs) > 1:
            # Calcola pairwise distances (cosine)
            dists = 1 - np.dot(patient_embs, patient_embs.T)
            # Upper triangular (escludi diagonale)
            upper_indices = np.triu_indices(len(patient_embs), k=1)
            intra_distances.extend(dists[upper_indices])

    # Inter-patient distances
    for i, p1 in enumerate(unique_patients):
        for p2 in unique_patients[i + 1:]:
            mask1 = patient_ids == p1
            mask2 = patient_ids == p2

            embs1 = embeddings[mask1]
            embs2 = embeddings[mask2]

            # Calcola pairwise distances
            dists = 1 - np.dot(embs1, embs2.T)
            inter_distances.extend(dists.flatten())

    # Medie
    d_intra = np.mean(intra_distances) if intra_distances else 0.0
    d_inter = np.mean(inter_distances) if inter_distances else 0.0

    metrics['d_intra'] = float(d_intra)
    metrics['d_inter'] = float(d_inter)

    # Clustering metrics
    try:
        db_score = davies_bouldin_score(embeddings, patient_ids)
        metrics['db'] = float(db_score)
    except Exception as e:
        logger.warning(f"Error computing DB for batch: {e}")
        metrics['db'] = float('inf')

    try:
        ch_score = calinski_harabasz_score(embeddings, patient_ids)
        metrics['ch'] = float(ch_score)
    except Exception as e:
        logger.warning(f"Error computing CH for batch: {e}")
        metrics['ch'] = 0.0

    return metrics


def compute_clustering_metrics(embeddings: np.ndarray, patient_ids: np.ndarray,
                               sample_size: int = None, compute_silhouette: bool = False,
                               compute_bw_ratio: bool = True, compute_db: bool = True) -> Dict:
    """
    Calcola le metriche di clustering principali.

    Args:
        embeddings: (N, D) array di embeddings
        patient_ids: (N,) array di patient IDs
        sample_size: se non None, campiona sample_size pazienti per Silhouette/DB (costoso)
        compute_silhouette: se False, salta il calcolo della Silhouette (lentissimo)
        compute_bw_ratio: se False, salta il calcolo del Between-Within Ratio
        compute_db: se False, salta il calcolo di Davies-Bouldin Index (costoso per grandi dataset)

    Returns:
        dict con metriche:
        - calinski_harabasz: float (higher is better)
        - davies_bouldin: float (lower is better) - opzionale
        - silhouette: float (range [-1, 1], higher is better) - opzionale
        - between_within_ratio: float (higher is better) - opzionale
    """
    metrics = {}

    try:
        # 1. Calinski-Harabasz Index (veloce)
        print(f"  Computing Calinski-Harabasz score...")
        ch_score = calinski_harabasz_score(embeddings, patient_ids)
        metrics['calinski_harabasz'] = float(ch_score)
        logger.info(f"CH Score: {ch_score:.4f}")
    except Exception as e:
        logger.warning(f"Error computing CH score: {e}")
        metrics['calinski_harabasz'] = 0.0

    try:
        # 2. Davies-Bouldin Index (opzionale per ottimizzazione training)
        if compute_db:
            print(f"  Computing Davies-Bouldin score...")
            db_score = davies_bouldin_score(embeddings, patient_ids)
            metrics['davies_bouldin'] = float(db_score)
            logger.info(f"DB Score: {db_score:.4f}")
        else:
            # Se disabilitato: set a default
            metrics['davies_bouldin'] = float('inf')
    except Exception as e:
        logger.warning(f"Error computing DB score: {e}")
        metrics['davies_bouldin'] = float('inf')

    try:
        # 3. Silhouette Coefficient (costoso - opzionale)
        if compute_silhouette:
            print(f"  Computing Silhouette score (this may take a while)...")
            if sample_size is not None and len(embeddings) > sample_size:
                sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                embeddings_sampled = embeddings[sample_indices]
                labels_sampled = patient_ids[sample_indices]
                silhouette = silhouette_score(embeddings_sampled, labels_sampled, sample_size=1000)
            else:
                silhouette = silhouette_score(embeddings, patient_ids, sample_size=1000)
            metrics['silhouette'] = float(silhouette)
            logger.info(f"Silhouette: {silhouette:.4f}")
        else:
            metrics['silhouette'] = None
    except Exception as e:
        logger.warning(f"Error computing Silhouette: {e}")
        metrics['silhouette'] = -1.0

    try:
        # 4. Between-Within Ratio (opzionale)
        if compute_bw_ratio:
            print(f"  Computing Between-Within Ratio...")
            bw_ratio = compute_between_within_ratio(embeddings, patient_ids)
            metrics['between_within_ratio'] = float(bw_ratio)
            logger.info(f"BW Ratio: {bw_ratio:.4f}")
        else:
            metrics['between_within_ratio'] = None
    except Exception as e:
        logger.warning(f"Error computing BW ratio: {e}")
        metrics['between_within_ratio'] = 0.0

    return metrics


def compute_between_within_ratio(embeddings: np.ndarray, patient_ids: np.ndarray) -> float:
    """
    Calcola Between-Within Ratio: media distanze inter-paziente / media distanze intra-paziente

    Higher is better (clusters più separati e compatti)
    """
    unique_patients = np.unique(patient_ids)

    # Intra-patient distances (within clusters)
    within_distances = []
    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_embeddings = embeddings[mask]

        if len(patient_embeddings) > 1:
            # Calcola pairwise distances
            dists = np.linalg.norm(patient_embeddings[:, np.newaxis] - patient_embeddings[np.newaxis, :], axis=2)
            # Media delle distanze upper-triangular (escludendo diagonale)
            upper_indices = np.triu_indices(len(patient_embeddings), k=1)
            within_distances.extend(dists[upper_indices])

    mean_within = np.mean(within_distances) if within_distances else 1.0

    # Inter-patient distances (between clusters)
    between_distances = []
    for i, p1 in enumerate(unique_patients):
        for p2 in unique_patients[i + 1:]:
            mask1 = patient_ids == p1
            mask2 = patient_ids == p2

            embs1 = embeddings[mask1]
            embs2 = embeddings[mask2]

            # Media distanze tra pazienti p1 e p2
            dists = np.linalg.norm(embs1[:, np.newaxis] - embs2[np.newaxis, :], axis=2)
            between_distances.extend(dists.flatten())

    mean_between = np.mean(between_distances) if between_distances else 1.0

    ratio = mean_between / mean_within if mean_within > 0 else 0.0
    return ratio


def compute_intra_inter_distances(embeddings: np.ndarray, patient_ids: np.ndarray) -> Dict:
    """
    Calcola intra-patient e inter-patient distances statistiche.

    Returns:
        dict con:
        - intra_patient_mean: media distanze intra-paziente
        - intra_patient_std: std distanze intra-paziente
        - inter_patient_mean: media distanze inter-paziente
        - inter_patient_std: std distanze inter-paziente
        - ratio: inter/intra
    """
    unique_patients = np.unique(patient_ids)

    # Intra-patient distances
    within_distances = []
    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_embeddings = embeddings[mask]

        if len(patient_embeddings) > 1:
            dists = np.linalg.norm(patient_embeddings[:, np.newaxis] - patient_embeddings[np.newaxis, :], axis=2)
            upper_indices = np.triu_indices(len(patient_embeddings), k=1)
            within_distances.extend(dists[upper_indices])

    # Inter-patient distances
    between_distances = []
    for i, p1 in enumerate(unique_patients):
        for p2 in unique_patients[i + 1:]:
            mask1 = patient_ids == p1
            mask2 = patient_ids == p2

            embs1 = embeddings[mask1]
            embs2 = embeddings[mask2]

            dists = np.linalg.norm(embs1[:, np.newaxis] - embs2[np.newaxis, :], axis=2)
            between_distances.extend(dists.flatten())

    within_distances = np.array(within_distances)
    between_distances = np.array(between_distances)

    return {
        'intra_patient_mean': float(np.mean(within_distances)),
        'intra_patient_std': float(np.std(within_distances)),
        'inter_patient_mean': float(np.mean(between_distances)),
        'inter_patient_std': float(np.std(between_distances)),
        'ratio': float(np.mean(between_distances) / (np.mean(within_distances) + 1e-8))
    }


def evaluate_embeddings(embeddings: np.ndarray, patient_ids: np.ndarray) -> Dict:
    """
    Valutazione completa degli embeddings.
    Calcola tutte le metriche.
    """
    results = {}

    # Clustering metrics
    results['clustering'] = compute_clustering_metrics(embeddings, patient_ids)

    # Intra/inter distances
    results['distances'] = compute_intra_inter_distances(embeddings, patient_ids)

    return results
