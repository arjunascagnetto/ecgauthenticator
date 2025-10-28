# Metodologia: Analisi di Similarità Intra-Paziente mediante Manhattan Distance

## Obiettivo

Determinare se gli esami ripetuti dello stesso paziente sono più simili tra loro rispetto agli esami di pazienti diversi. In altre parole, valutare se ogni paziente possiede una "firma ECG" distintiva nel dataset.

## Perché Manhattan Distance?

### Scelta della Metrica

La distanza Manhattan (L1 norm) è stata selezionata per le seguenti ragioni:

1. **Robustezza agli Outlier**: La Manhattan distance utilizza differenze assolute anziché quadratiche. Questo riduce significativamente l'impatto dei valori anomali, comuni nei dati medici [1], [2].

2. **Performance in Spazi High-Dimensional**: Con 13 features ECG, il dataset è relativamente high-dimensional. La ricerca dimostra che la Manhattan distance supera la distanza euclidea in tali contesti, mentre euclidea soffre del problema di concentrazione della distanza [2], [3].

3. **Efficienza Computazionale**: A scala (164K esami), Manhattan offre complessità O(n) semplice, fondamentale per dataset di grandi dimensioni [1], [2].

4. **Adeguatezza ai Dati Medici**: Studi su dati clinici e di pianificazione dei servizi sanitari confermano che Manhattan fornisce stime più affidabili di euclidea [1].

## Perché NOT il Rapporto CV_within/CV_between?

### Limitazioni del Coefficient of Variation

Il rapporto CV (Coefficient of Variation intra vs inter) non risponde adeguatamente alla nostra domanda per i seguenti motivi:

1. **Perdita di Informazione Strutturale**: Il CV riduce i dati a singoli numeri (media e varianza), perdendo completamente la struttura di similarità tra coppie di esami [11], [12].

2. **Insensibilità ai Pattern Anomali**: Il CV non cattura se gli esami dello stesso paziente sono effettivamente simili, bensì solo se la loro variazione è piccola. Due distribuzioni con stessa varianza ma pattern di similarità completamente diversi avrebbero stesso CV [11].

3. **Inadeguato per Relazioni Patient-Centric**: L'analisi CV partiziona la varianza globale, non le relazioni dirette tra esami dello stesso paziente versus esami di pazienti diversi [8], [15].

4. **Inaccessibilità a Metriche di Distanza Pairwise**: Solo le distanze pairwise permettono di quantificare concretamente se d(esame_i paziente A, esame_j paziente A) < d(esame_i paziente A, esame_k paziente B) [6], [7].

## Metodologia

### 1. Calcolo Distanze Intra-Paziente (Within)

Per ogni paziente con ≥2 esami:
- Calcola tutte le distanze Manhattan pairwise tra gli esami
- Raccogli i valori nel vettore `intra_distances`
- Media: fornisce la "compattezza" degli esami dello stesso paziente

### 2. Calcolo Distanze Inter-Paziente (Between)

- Seleziona coppie di pazienti diversi
- Calcola distanze Manhattan tra tutti gli esami di ciascuna coppia
- Campionamento intelligente (max 100 pazienti per paziente) per efficienza su 53K pazienti
- Media: fornisce la "separazione" tra pazienti

### 3. Rapporto Intra/Inter

```
ratio = mean(intra_distances) / mean(inter_distances)
```

**Interpretazione**:
- **ratio < 0.5**: Esami dello stesso paziente MOLTO PIÙ simili (firma ECG distintiva)
- **ratio 0.5-1.0**: Esami del paziente comparabilmente/leggermente più simili
- **ratio 1.0-2.0**: Variabilità intra ≈ variabilità inter (scarsa separazione)
- **ratio > 2.0**: Esami dello stesso paziente MENO simili tra loro (problema critico)

## Validazione della Scelta

Studi sulla similarità paziente-specifica in medicina hanno dimostrato che:

1. Modelli personalizzati basati sulla similarità paziente hanno performance significativamente superiori ai modelli di popolazione [6], [7].

2. L'identificazione di pazienti simili è un compito fondamentale in medicina personalizzata, e le metriche di distanza rimangono il metodo più consolidato [5].

3. Quando i pazienti sono raggruppati correttamente per similarità, la eterogeneità nei fattori di rischio diventa evidente anche in coorti diagnosticamente identiche [7].

4. La proper partizione tra varianza within-subject e between-subject è critica per valide inferenze statistiche [8], [15].

## Dataset

- **Esami**: 164.440 (dopo filtro pazienti con singolo esame)
- **Pazienti**: 53.079
- **Features ECG**: 13 (VentricularRate, PRInterval, QRSDuration, QTInterval, QTCorrected, PAxis, RAxis, TAxis, QOnset, QOffset, POnset, POffset, TOffset)
- **Metrica Distanza**: Manhattan (L1 norm)

## Output

Due file CSV:
1. `manhattan_distance_analysis.csv`: statistiche riassuntive (medie, std, rapporto)
2. `manhattan_distances_distribution.csv`: tutte le distanze calcolate (per analisi ulteriore)

---

## Riferimenti

[1] R. Shahid, S. Bertazzon, M. L. Knudtson, and W. A. Ghali, "Comparison of distance measures in spatial analytical modeling for health service planning," *BMC Health Services Research*, vol. 9, no. 200, 2009. DOI: 10.1186/1472-6963-9-200

[2] C. C. Aggarwal, A. Hinneburg, and D. A. Keim, "On the Surprising Behavior of Distance Metrics in High Dimensional Space," in *Proc. 8th Int. Conf. Database Theory (ICDT)*, London, UK, 2001, pp. 420-434. DOI: 10.1007/3-540-44503-X_27

[3] D. Peng, Z. Gui, and H. Wu, "Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect," *arXiv:2401.00422* [cs.LG], 2025. DOI: 10.48550/arXiv.2401.00422

[5] A. Sharafoddini, J. A. Dubin, and J. Lee, "Patient Similarity in Prediction Models Based on Health Data: A Scoping Review," *JMIR Medical Informatics*, vol. 5, no. 1, p. e7, Mar. 2017. DOI: 10.2196/medinform.6730

[6] N. Wang, Y. Huang, H. Liu, X. Fei, L. Wei, X. Zhao, and H. Chen, "Measurement and application of patient similarity in personalized predictive modeling based on electronic medical records," *BioMedical Engineering OnLine*, vol. 18, no. 1, art. 98, 2019. DOI: 10.1186/s12938-019-0718-2

[7] K. Ng, J. Sun, J. Hu, and F. Wang, "Personalized Predictive Modeling and Risk Factor Identification using Patient Similarity," *AMIA Jt Summits Transl Sci Proc.*, vol. 2015, pp. 132-136, Mar. 2015. PMCID: PMC4525240

[8] P. Schober and T. R. Vetter, "Repeated measures designs and analysis of longitudinal data: if at first you do not succeed—try, try again," *Anesth. Analg.*, vol. 127, no. 2, pp. 569-575, Aug. 2018. DOI: 10.1213/ANE.0000000000003511

[11] G. F. Reed, F. Lynn, and B. D. Meade, "Use of Coefficient of Variation in Assessing Variability of Quantitative Assays," *Clinical and Diagnostic Laboratory Immunology*, vol. 9, no. 6, pp. 1235-1239, Nov. 2002. DOI: 10.1128/CDLI.9.6.1235-1239.2002

[12] G. Coste and F. Lemaitre, "The Role of Intra-Patient Variability of Tacrolimus Drug Concentrations in Solid Organ Transplantation: A Focus on Liver, Heart, Lung and Pancreas," *Pharmaceutics*, vol. 14, no. 2, art. 379, Feb. 2022. DOI: 10.3390/pharmaceutics14020379

[15] M. Wang, "Generalized Estimating Equations in Longitudinal Data Analysis: A Review and Recent Developments," *Advances in Statistics*, vol. 2014, art. 303728, 2014. DOI: 10.1155/2014/303728
