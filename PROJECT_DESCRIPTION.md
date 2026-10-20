# Adaptive GPU-Accelerated Mini-Batch K-Means for Satellite Image Clustering

**Project brief and research foundation — Version 1.0, 18 October 2026**  
**Area:** Parallel Computing / GPU Computing / Unsupervised Satellite-Image Analysis  
**Status:** Proposed work; no experimental results have yet been produced by this project.

## 1. Abstract

Multispectral satellite imagery contains millions of pixels, each represented by measurements from several spectral bands. Standard K-Means repeatedly assigns data points to their nearest centroids and updates those centroids, making large workloads computationally expensive. This project proposes a Python-based GPU clustering framework that accelerates K-Means using explicitly programmed CUDA kernels, investigates Mini-Batch updates and approximate nearest-centroid (ANN, *Approximate Nearest Neighbor*) assignment, and tests a convergence-aware controller that adjusts both Mini-Batch size and ANN search effort during execution. Sentinel-2 Level-2A imagery will supply realistic multispectral workloads. Sequential CPU, exact GPU, fixed-parameter GPU and adaptive GPU variants will be benchmarked using wall-clock runtime, kernel runtime, memory consumption, assignment/search work, clustering objective and quality metrics. The study will use repeated random seeds, held-out satellite scenes, matched initializations and ablation tests. The objective is to determine **whether, when and at what quality cost** jointly adapting the two optimizations provides an advantage; speedup is not assumed in advance.

## 2. Problem statement

K-Means with N samples, K centroids and D spectral features typically requires distance computations proportional to N × K × D per full assignment pass. A large Sentinel-2 crop may contain millions of valid pixels. CUDA can parallelize distance and centroid-update computations, but GPU memory transfers, aggregation and repeated unnecessary distance comparisons can still dominate runtime. Mini-Batch K-Means processes a smaller sample per update; ANN reduces the effort used to locate a nearby centroid. Both alter computation and can affect accuracy, convergence and overhead. Their relative usefulness may change as centroids stabilize. It is not established from the selected literature that **joint, convergence-driven adaptation** of both controls is advantageous for low-dimensional multispectral satellite imagery and realistic cluster counts.

## 3. Proposed solution

1. Preprocess selected Sentinel-2 Level-2A scenes into valid, aligned, normalized float32 pixel feature arrays.
2. Implement one sequential compiled CPU K-Means reference and one **custom CUDA K-Means** implementation in Python (Numba-CUDA or a compatible Python-CUDA kernel compiler).
3. Implement GPU Mini-Batch K-Means with persisted running centroid statistics (not Mini-Batch Gradient Descent).
4. Prototype genuinely approximate nearest-centroid assignment and evaluate exact-assignment errors, index maintenance and transfer overhead against exact CUDA assignment.
5. Implement a controller using periodically measured centroid displacement and/or changes in assignment on a fixed monitoring sample. It can increase, maintain or reduce batch size and ANN effort within documented bounds; the direction and thresholds must be experimentally chosen, not assumed.
6. Run separate batch-adaptive, ANN-adaptive and jointly adaptive ablations; compare speed and quality with fixed settings.

**Expected deliverable:** reproducible source code, a documented pipeline, unit and integration tests, experimental result tables/figures, clustered-image visualizations, and a limitations-aware report. **Not a deliverable:** pretrained land-cover classification labels or a guaranteed faster algorithm.

## 4. Dataset and environment

- **Primary open data:** Copernicus Sentinel-2 Level-2A surface-reflectance imagery: https://browser.dataspace.copernicus.eu/
- **Initial 10 m bands:** B02 (blue), B03 (green), B04 (red), B08 (near infrared). Optional additional 20 m bands require spatial resampling onto a common grid.
- **Quality mask:** use the Sentinel-2 Scene Classification Layer (SCL), with nearest-neighbor resampling for category values; exclude no-data/defective pixels, cloud shadows and cloud/cirrus classes. Define any snow exclusion explicitly.
- **Scaling:** convert reflectance bands to a common documented floating-point scale, derive training-scene normalization statistics and use them unchanged for the held-out scenes.
- **Evaluation:** at least one scene group for development/tuning and different scene(s) for final testing; document scene product IDs, acquisition dates, region, cloud fraction, cropped dimensions, bands and valid-pixel counts.
- **Available development system:** Windows 11; Ubuntu 24.04 WSL2; Intel Core i5-13450HX (10 cores/16 logical processors); NVIDIA RTX 4050 Laptop GPU (6 GB VRAM); 24 GB RAM; CUDA Toolkit 12.0 detected under WSL2. Actual Python kernel execution and Python-package compatibility remain to be verified.

## 5. Literature review (verified primary references)

| Work | Central idea and reported findings | Implication and difference from proposed study |
| --- | --- | --- |
| **Pant, Leigh & Fajardo Rueda (2025)** | Compares eight parallelized K-Means variants for large satellite image data; explores initialization, outlier handling and reducing repeated nearest-neighbor calculations. | Strong direct satellite-clustering motivation. Does not demonstrate the proposed joint convergence-controlled GPU Mini-Batch + ANN framework. |
| **Han & Lee (2024)** | Inter-image K-Means on PlanetScope imagery with multicore CPU/GPU implementations; reports up to **12.83×** multicore, **25.53×** GPU and **39.00×** GPU with parallel reduction relative to its sequential baseline. | Shows value of GPU parallelization and efficient reductions. Different clustering objective and no joint Mini-Batch/ANN adaptation. |
| **Li, Frank & Pfahringer (2023)** | Asynchronous Selective Batched (ASB) exact GPU K-Means using triangle-inequality pruning and CPU/GPU overlap; reports up to **15×** faster than standard GPU K-Means on tested workloads. | Batching and reducing transfers are established techniques. **Selective batching in exact K-Means is not equivalent to Mini-Batch centroid updates.** |
| **Spalding-Jamieson, Robson & Zheng (2025)** | Seeded approximate nearest-neighbor methods for K-Means at very large K and high D; optimized approximate centroid searches rather than naive ANN insertion. | Closest to the ANN assignment component. Its very high cluster-count/high-dimensional regime differs from small-K Sentinel-2 clustering. Moving-centroid index maintenance must be considered. |
| **Zhu et al. (2024)** | Staleness-reduction Mini-Batch K-Means, designed to address noisy or stale centroid information; reports **40–130×** faster convergence to a target loss than ordinary Mini-Batch K-Means and **0.2%–1.7%** lower final loss under its evaluation. | Demonstrates that fewer examples per update can hurt convergence; validates the need for quality-constrained adaptation. |
| **Zhang et al. (2021)** | Satellite-image time-series clustering using time-adaptive optimal transport and scikit-learn Mini-Batch K-Means. | Establishes Mini-Batch K-Means use in remote sensing; main innovation was time-series similarity, not CUDA/ANN. |

**Important comparison warning:** these reported numbers come from different hardware, tasks, datasets and baselines. They cannot be combined into one cross-paper leaderboard or treated as predicted speedups for this project.

## 6. Research gap and novelty statement

The reviewed literature establishes (i) GPU-accelerated K-Means; (ii) batching, selective data movement and exact distance pruning; (iii) Mini-Batch updates; and (iv) specialized approximate nearest-centroid assignments. However, the papers in this review **do not demonstrate** a controlled study of *jointly* adapting **Mini-Batch update size and ANN nearest-centroid search effort according to online convergence measurements**, evaluated on real Sentinel-2 pixel data against individually adaptive variants with GPU overhead included.

This is a **provisional research gap based on the reviewed papers**, not a claim that nobody anywhere has done similar work. A targeted literature check should be repeated before final submission.

**Proposed contributions** (to be assessed, not claimed as completed):
1. Custom Python-based CUDA parallel K-Means for reproducible satellite workloads.
2. Separate fixed and adaptive GPU Mini-Batch and ANN implementations.
3. Measured controller policy balancing runtime, assignment accuracy and final clustering quality.
4. Ablation tests that identify when GPU exact search, ANN, Mini-Batching and joint adaptation help or fail across image sizes and K.

## 7. Research questions and hypotheses

- **RQ1:** How much faster is the custom exact CUDA K-Means than an optimized single-worker CPU implementation on the same hardware, dataset and initialization?
- **RQ2:** How do fixed Mini-Batch sizes affect time-to-quality, peak GPU memory and final clustering objective?
- **RQ3:** At which values of K, if any, does approximate nearest-centroid search beat exact CUDA assignment after including index updates and transfers?
- **RQ4:** Does adaptation of batch size alone, ANN effort alone, or both together deliver better *time-to-a-target-quality* than fixed methods?
- **RQ5:** Are the results robust to random seeds, different Sentinel-2 scenes and larger input arrays?

A reasonable hypothesis is that GPU parallelization will improve throughput on sufficiently large inputs. **No directional claim is made that decreasing both adaptation knobs near convergence is optimal.**

## 8. Experimental methodology and acceptance criteria

- Use identical preprocessed arrays, data splits, K and initial centroids for corresponding comparisons; record all random seeds.
- Test at least three seeds in development; increase repeat counts for final comparisons as runtime permits. Use multiple repeated timed runs once warmed up.
- Keep data initialization, GPU transfer, algorithm computation and index maintenance timings separately visible; report both total wall time and GPU compute time.
- Compare **the same achieved quality or a documented quality threshold**, not only the number of K-Means iterations, since a Mini-Batch iteration handles fewer pixels.
- Main objective: full-dataset within-cluster sum of squared distances (inertia/SSE), evaluated at controlled intervals; record relative objective gap to an exact reference when useful.
- Additional sampled metrics: Silhouette Score and Davies–Bouldin Index. Report assignment agreement/ANN recall against exact assignments on fixed probe samples; do not confuse agreement with true land-cover accuracy.
- For any manual land-cover interpretation, say **clusters** rather than **land-cover classes**, unless verified with aligned labeled reference data.
- Report time, speedup, K, pixel count, number of bands, memory, GPU utilization where measurable, number of distance evaluations, index rebuild/update time and quality for all variants.

**Outcome considered valuable even when negative:** clear evidence that exact CUDA assignment outperforms ANN at realistic small K, or that some adaptive schedules degrade clusters. Report the conditions and reasons accurately.

## 9. Scope decision requiring confirmation

The teacher's original proposal calls for sequential, OpenMP and CUDA baselines and suggests C/C++. For the **initial solo implementation**, the team currently plans to use Python for CPU reference and custom CUDA kernels and to **defer OpenMP**. This does **not** remove the teacher's OpenMP requirement. Seek approval before presenting the narrowed implementation as complete fulfillment of the approved proposal; otherwise budget an OpenMP add-on as a later milestone.

## 10. Primary bibliography

1. Pant, Y. R., Leigh, L., & Fajardo Rueda, J. (2025). *Improving K-Means Clustering: A Comparative Study of Parallelized Version of Modified K-Means Algorithm for Clustering of Satellite Images*. Algorithms, 18(8), 532. https://doi.org/10.3390/a18080532
2. Han, S., & Lee, J. (2024). *Parallelized Inter-Image k-Means Clustering Algorithm for Unsupervised Classification of Series of Satellite Images*. Remote Sensing, 16(1), 102. https://doi.org/10.3390/rs16010102
3. Li, M., Frank, E., & Pfahringer, B. (2023). *Large scale K-means clustering using GPUs*. Data Mining and Knowledge Discovery, 37, 67–109. https://doi.org/10.1007/s10618-022-00869-6
4. Spalding-Jamieson, J., Robson, E. W., & Zheng, D. W. (2025). *Scalable k-Means Clustering for Large k via Seeded Approximate Nearest-Neighbor Search*. arXiv:2502.06163. https://arxiv.org/abs/2502.06163
5. Zhu, X., Sun, J., He, Z., Jiang, J., & Wang, Z. (2024). *Staleness-Reduction Mini-Batch K-Means*. IEEE Transactions on Neural Networks and Learning Systems, 35(10), 14424–14436. https://doi.org/10.1109/TNNLS.2023.3279122
6. Zhang, Z., Tang, P., Zhang, W., & Tang, L. (2021). *Satellite Image Time Series Clustering via Time Adaptive Optimal Transport*. Remote Sensing, 13(19), 3993. https://doi.org/10.3390/rs13193993

**Source distinction:** The core objective, proposed components and teacher concerns originate from the supplied project proposal and its comments. The above publication details, research summaries and measured results come from the cited papers. The implementation choices, milestones, dataset strategy and acceptance criteria are the current proposed plan, not established findings.
