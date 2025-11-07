# Metric Learning Pair Sampling Strategies for Stable ECG Embeddings

## 1. Contesto e Obiettivo
- Task: apprendere embedding ECG paziente-specifici (13 features standardizzate, piu esami per paziente) massimizzando vicinanza intra-paziente e separazione inter-paziente.
- Criticita: squilibrio (pochi positivi per paziente, molti negativi), rumore fisiologico (variabilita intra), rischio di collasso dell'embedding se sampling/mining troppo aggressivo.
- Target: selezionare combinazioni `loss + strategia di campionatura` che preservino stabilita del training nei contesti clinici.

## 2. Loss per Metric Learning e Sensibilita al Campionamento

| Loss | Segnale Principale | Sensibilita al Mining | Note operative su ECG |
|------|-------------------|------------------------|------------------------|
| Contrastive (Hadsell et al. 2006) | Forza distanza intra << inter con margine fisso | Alta: serve mix di coppie easy/medium; hard-only causa collasso | Funziona bene con batch bilanciati PK; aumentare progressivamente margine per gestire rumore ECG |
| Triplet (Schroff et al. 2015; Hermans et al. 2017) | Minimizza `d(anchor,pos) + margin < d(anchor,neg)` | Molto dipendente da mining (semi-hard raccomandato) | Usare PK sampling con >=4 esempi/paziente; margin scheduling per soggetti rumorosi |
| Lifted Structured (Oh Song et al. 2016) | Loss su tutte le coppie in batch con pesi log-sum-exp | Altissima: batch troppo easy -> gradiente debole | Richiede batch grandi (>=64) per catturare variabilita ECG |
| N-Pair / Multi-class (Sohn 2016) | Classifica rispetto a un negativo per classe | Media: riduce necessita di hard mining, ma serve class balance | Consigliato quando #pazienti per minibatch e alto ma #esami/paziente limitato |
| Histogram (Ustinova & Lempitsky 2016) | Allinea distribuzioni di distanze intra/inter | Bassa: lavora con tutte le coppie del batch | Robusta al noise ECG, ma necessita stimare istogrammi stabili (batch>=128) |
| Multi-Similarity (Wang et al. 2019) | Pesa molteplici relazioni con soft weighting | Media: integra pesi adattivi, riduce hard mining manuale | Ottimo compromesso per ECG: combina easy/medium/hard in un'unica loss |
| Circle Loss (Wang et al. 2020) | Ottimizza similitudini con fattori auto-adattivi | Media: minore rischio di collasso grazie a pesi dinamici | Funziona bene con normalizzazione L2 e temperature moderate (gamma circa 80) |
| Proxy-NCA (Movshovitz-Attias et al. 2017) | Sostituisce negativi con proxy per classe | Bassa: niente mining esplicito | Utile quando pochi esami/paziente; necessita buona inizializzazione dei proxy |
| Proxy-Anchor (Kim et al. 2020) | Proxy con margini dinamici e pesi adattivi | Bassa: gestisce outlier tramite weight clipping | Adatto a dataset grandi con pazienti sbilanciati |
| SoftTriple (Qian et al. 2019) | Multi-centro per classe all'interno di softmax | Media: combina proxy e cluster intra-classe | Modella piu stati ECG per paziente (riposo/sforzo) senza mining manuale |
| InfoNCE / NT-Xent (Chen et al. 2020) | Contrastivo normalizzato con temperatura | Richiede molte viste augmentate | Applicabile se si generano augmentazioni ECG (rumore, warping) + memory queue |
| Large-margin classification (ArcFace / CosFace) | Loss angolare supervisionata | Moderata: richiede batch bilanciati | Utile come classificatore di supporto per valutare separazione embedding |

**Osservazione chiave:** le loss con pesi adattivi (Multi-Similarity, Circle, Proxy-Anchor) mitigano il rischio di collasso perche combinano contributi da coppie easy e hard senza introdurre gradienti estremi.

## 3. Tecniche di Campionatura e Mining di Coppie

### 3.1 Costruzione del Mini-batch
- **PK Sampling (Kaya & Bilge 2019):** seleziona `P` pazienti x `K` esami ciascuno (tipicamente 8x4). Garantisce sufficiente varieta di positivi/negativi nello stesso batch, base per tutte le strategie successive.
- **Stratificazione per metadati ECG:** bilancia esami per genere, range QT, condizioni cliniche; riduce bias e stabilizza margini.

### 3.2 Strategie di Mining
1. **Random / Batch-All:** usa tutte le coppie possibili nel batch (Lifted, Histogram). Basso rischio collasso ma gradiente diluito se molte coppie easy.
2. **Semi-Hard Mining (Schroff et al. 2015):** seleziona negativi con `d_pos < d_neg < d_pos + margin`. Equilibrio tra informativita e stabilita; ideale in warm-up per ECG rumorosi.
3. **Batch-Hard (Hermans et al. 2017):** per ogni anchor sceglie positivo piu lontano e negativo piu vicino nel batch. Massimizza segnale ma aumenta rischio collasso se usato dall'inizio, introdurlo dopo 5-10 epoche.
4. **Distance-Weighted Sampling (Wu et al. 2017):** campiona negativi con probabilita proporzionale a distanza^(d-2), evitando concentrazione su negativi troppo facili o troppo difficili. Efficace in alta dimensione (64D) e stabile per ECG normali.
5. **Smart o Hierarchical Mining (Harwood et al. 2017; Ge et al. 2018):** cascata che filtra negativi partendo da easy verso hard; utile quando il numero di pazienti supera 100k perche riduce il costo quadratico.
6. **Curriculum o Self-Paced Mining:** aumenta progressivamente la durezza (margine o peso hard negatives) in funzione dell'epoca. Migliora stabilita dopo warm-up; implementabile con scheduler sul parametro beta (vedi appendice nel file `CONTRASTIVE_LOSS_PAIR_SELECTION.md`).
7. **Memory Bank o Queue (He et al. 2020; Chen et al. 2020):** mantiene code di embedding per campionare negativi extra-batch. In supervisione, aggiornare la queue con media mobile per evitare stale features; attenzione a drift ECG, limitare la lunghezza a circa 4096 elementi.
8. **Proxy-based Sampling (Movshovitz-Attias et al. 2017; Kim et al. 2020):** riduce il mining esplicito utilizzando rappresentanti per paziente o cluster. Adatto quando gli esami per paziente sono pochi (<5) ma il numero di pazienti e alto.
9. **Variance-based Filtering (Musgrave et al. 2020):** penalizza selezione di negativi con distanza < epsilon per molte epoche (indicatore di rumore) per evitare oscillazioni di gradiente.

### 3.3 Heuristics Specifiche ECG
- **Finestre temporali:** trattare esami dello stesso paziente separati da molti giorni come positivi "hard" per modellare variazione fisiologica.
- **Augmentazioni controllate:** jitter di intervalli, scaling amplitude, rumore bianco per generare viste positive supplementari in InfoNCE / NT-Xent mantenendo significato clinico.
- **Pesatura per qualita segnale:** escludere esami con rumore elevato (flag `InRange`) dai negativi estremi per evitare gradienti spurii.

## 4. Stabilita e Mitigazione del Collasso
- **Warm-up multi-fase:** iniziare con contrastive o NT-Xent piu sampling casuale, introdurre semi-hard quando la loss converge, poi batch-hard o distance-weighted per gli ultimi stadi.
- **Normalizzazione L2 e temperature controllate:** fissare `||embedding||_2 = 1` e usare temperature 0.05-0.1 (InfoNCE) o margini 0.4-0.7 (triplet) per limitare gradienti esplosivi.
- **Regolarizzazione di varianza:** aggiungere termini tipo Center Loss o Batch Variance Loss per mantenere ampiezza dello spazio latente.
- **Proxy piu pair mining ibrido:** combinare Proxy-Anchor (stabile) con mining leggeri su residui per ridurre collasso quando alcuni pazienti hanno pochi esami.
- **Monitoraggio metriche:** tracciare rapporto intra/inter, deviazione standard delle distanze e rank-1 accuracy; sospendere mining duro se la varianza embedding scende sotto soglia definita.

## 5. Pipeline Raccomandata per il Dataset ECG
1. **PK Sampling (P=12 pazienti, K=4 esami)** mantenendo distribuzione per sesso e range QT.
2. **Loss primaria:** Multi-Similarity (robusta) con Circle Loss come regolarizzatore angolare su embedding normalizzati.
3. **Fasi mining:** warm-up (5 epoche) con batch-all; epoche 6-20 distance-weighted piu semi-hard; successivamente batch-hard solo per il 20% dei batch piu difficili.
4. **Regolarizzazioni:** center loss (lambda=1e-3), gradient clipping (1.0), label smoothing 0.05.
5. **Validazione continua:** calcolo periodico del rapporto manhattan intra/inter, t-SNE su subset e monitor mAP/NMI sugli embedding 32D.

## 6. Riferimenti (BibTeX)

```bibtex
@inproceedings{hadsell2006dimensionality,
  title        = {Dimensionality Reduction by Learning an Invariant Mapping},
  author       = {Hadsell, Raia and Chopra, Sumit and LeCun, Yann},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2006}
}

@inproceedings{schroff2015facenet,
  title        = {FaceNet: A Unified Embedding for Face Recognition and Clustering},
  author       = {Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2015}
}

@inproceedings{hermans2017defense,
  title        = {In Defense of the Triplet Loss for Person Re-Identification},
  author       = {Hermans, Alexander and Beyer, Lucas and Leibe, Bastian},
  booktitle    = {arXiv preprint arXiv:1703.07737},
  year         = {2017}
}

@inproceedings{ohsong2016deep,
  title        = {Deep Metric Learning via Lifted Structured Feature Embedding},
  author       = {Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2016}
}

@inproceedings{sohn2016improved,
  title        = {Improved Deep Metric Learning with Multi-Class N-Pair Loss Objective},
  author       = {Sohn, Kihyuk},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2016}
}

@inproceedings{ustinova2016learning,
  title        = {Learning Deep Embeddings with Histogram Loss},
  author       = {Ustinova, Evgeniya and Lempitsky, Victor},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2016}
}

@inproceedings{movshovitz2017nofuss,
  title        = {No Fuss Distance Metric Learning Using Proxies},
  author       = {Movshovitz-Attias, Yair and Toshev, Alexander and Leung, Thomas and Ioffe, Sergey and Singh, Saurabh},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year         = {2017}
}

@inproceedings{wu2017sampling,
  title        = {Sampling Matters in Deep Embedding Learning},
  author       = {Wu, Chao-Yuan and Manmatha, Raghuraman and Smola, Alexander and Krahenbuhl, Philipp},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year         = {2017}
}

@inproceedings{harwood2017smart,
  title        = {Smart Mining for Deep Metric Learning},
  author       = {Harwood, Ben and Kumar, Vijay and Carneiro, Gustavo and Reid, Ian and Drummond, Tom},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year         = {2017}
}

@inproceedings{ge2018deep,
  title        = {Deep Metric Learning with Hierarchical Triplet Loss},
  author       = {Ge, Weifeng and Zhu, Yizhou},
  booktitle    = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year         = {2018}
}

@inproceedings{wang2019multisimilarity,
  title        = {Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
  author       = {Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R.},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2019}
}

@inproceedings{kim2020proxy,
  title        = {Proxy Anchor Loss for Deep Metric Learning},
  author       = {Kim, Suhyeon and Kim, Dongwon and Cho, Minsu and Park, Jinwoo},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2020}
}

@inproceedings{wang2020circle,
  title        = {Circle Loss: A Unified Perspective of Pair Similarity Optimization},
  author       = {Wang, Yue and Sun, Xiyu and Liu, Biao and Xue, Yidong and Sun, Qi and Wang, Yu and Li, Zongben and Wang, Yuan},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2020}
}

@inproceedings{qian2019softtriple,
  title        = {SoftTriple Loss: Deep Metric Learning Without Triplet Sampling},
  author       = {Qian, Qi and Shang, Rui and Sun, Bo and Hu, Hao and Li, Juhua and Jin, Rong},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year         = {2019}
}

@article{musgrave2020metric,
  title        = {Metric Learning: The Key to Deep Metric Learning is Proper Objective, Sampling, and Distance},
  author       = {Musgrave, Kevin and Belongie, Serge and Lim, Ser-Nam},
  journal      = {International Journal of Computer Vision},
  year         = {2020},
  volume       = {128},
  number       = {10},
  pages        = {2761--2793}
}

@article{kaya2019deep,
  title        = {Deep Metric Learning: A Survey},
  author       = {Kaya, Mahmut and Bilge, Huseyin},
  journal      = {Symmetry},
  year         = {2019},
  volume       = {11},
  number       = {9},
  pages        = {1066}
}

@inproceedings{he2020momentum,
  title        = {Momentum Contrast for Unsupervised Visual Representation Learning},
  author       = {He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle    = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2020}
}

@inproceedings{chen2020simple,
  title        = {A Simple Framework for Contrastive Learning of Visual Representations},
  author       = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle    = {Proceedings of the International Conference on Machine Learning (ICML)},
  year         = {2020}
}

@article{biel2001ecg,
  title        = {ECG Analysis: A New Approach in Human Identification},
  author       = {Biel, Lena and Pettersson, Ola and Philipson, Lennart and Wide, Peter},
  journal      = {IEEE Transactions on Instrumentation and Measurement},
  year         = {2001},
  volume       = {50},
  number       = {3},
  pages        = {808--812}
}
```
