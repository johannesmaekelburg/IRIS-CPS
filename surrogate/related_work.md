# Related Work: Neural Models for Geometric Overlap Prediction

## Directly relevant: convex set intersection

- **Tajine & Elizondo (1996)** — "A neural network measuring the intersection of m-dimensional convex polyhedra." Analog/digital NN that detects and measures degree of intersection between convex polyhedra. Closest classical precedent to our problem — predicts scalar overlap between two convex sets.

- **Bao, He, Hirst et al. (2021)** — "Polytopes and Machine Learning" (arXiv:2109.09602). Feed-forward nets and autoencoders for predicting polytope properties (volume, reflexivity). Key finding: **representation matters more than architecture** — Plücker coordinates outperform vertex representations. Volume prediction is #P-hard but learnable from data.

## Set-based architectures

- **Zaheer et al. (2017)** — "Deep Sets" (NeurIPS). f(X) = ρ(Σ_i φ(x_i)). Provably universal for permutation-invariant functions. The canonical architecture for sets. Sum-pooling is the key operation.

- **Lee et al. (2019)** — "Set Transformer" (ICML, arXiv:1810.00825). Attention-based encoder for sets. Models pairwise interactions among set elements via self-attention. Inducing points reduce O(n²) to O(n). Cross-attention between two sets could capture which generators interact.

- **Qi et al. (2017)** — "PointNet" (CVPR, arXiv:1612.00593). f(X) = γ(max_i{h(x_i)}). Max-pooling selects critical points. For zonotope overlap, the bottleneck dimension (narrowest overlap axis) dominates — max-pool may outperform sum-pool.

## IoU prediction and differentiable overlap

- **Rezatofighi et al. (2019)** — "Generalized Intersection over Union" (CVPR). Standard IoU has zero gradient when shapes don't overlap. GIoU uses the smallest enclosing convex set to provide gradient signal. Relevant for training when zonotopes are fully disjoint.

- **Zheng et al. (2020)** — "Rotation-Robust IoU for 3D Object Detection" (ECCV). Handles the combinatorial complexity of convex intersection under rotation via projection operations. Relevant for non-axis-aligned zonotope overlap.

## Neural surrogates for reachability / zonotope computation

- **Solanki, Vertovec et al. (2025)** — "Certified Approximate Reachability (CARe)" (arXiv:2503.23912). Neural network approximating Hamilton-Jacobi PDE solutions for reachable sets with formal error bounds. Uses CEGIS for refinement.

- **Thorpe et al. (2020)** — "Learning Approximate Forward Reachable Sets Using Separating Kernels" (arXiv:2011.09678). Kernel-based classifier in RKHS for reachable set estimation. Frames the problem as support estimation — zonotope overlap can be framed as: does the Minkowski difference contain the origin?

- **Real-time MPC with Zonotope NNs (2024)** — arXiv:2403.16485. Cascaded NNs output zonotope parameters (center + generators). Collision detection via Minkowski sum + origin containment. Key lesson: networks can directly output zonotope geometry.

## Zonotope-NN connections

- **Hybrid Zonotopes ≡ ReLU Networks (2023)** — arXiv:2304.02755. Proves hybrid zonotopes can exactly represent ReLU NN input-output mappings. Suggests ReLU architectures have natural affinity for zonotope geometry.

- **Parameterized Hardness of Zonotope Containment (2025)** — arXiv:2509.22849. Containment is W[1]-hard w.r.t. dimension. Motivates learned surrogates — exact containment is fundamentally hard.

## Shape comparison architectures

- **CvxNet (Deng et al., 2020)** — "Learnable Convex Decomposition" (CVPR, arXiv:1909.05736). Represents convex shapes implicitly as half-space intersections (support functions). Differentiable and topology-agnostic.

- **OverlapNet (Chen et al., 2021)** — Autonomous Robots. Siamese network for predicting overlap between geometric observations. Encodes each observation independently through shared-weight branches, then combines for overlap prediction.

- **Differentiable Convex Optimization Layers (2024)** — arXiv:2412.20679. LP feasibility checks (like zonotope containment) can be embedded as differentiable neural network layers. Could serve as an inductive bias instead of learning overlap from scratch.

## Key architectural takeaways for our problem

1. **Feature/representation design > architecture choice** (Bao et al.)
2. **Permutation invariance is essential** for generator sets — DeepSets is minimal correct choice
3. **Max-pool captures bottleneck** (PointNet) — the tightest axis dominates inconsistency
4. **Siamese pattern** for comparing two shapes: encode independently, combine (OverlapNet)
5. **Cross-attention** between generator sets models which source generators interact with which target generators (Set Transformer)
6. **ReLU networks are naturally suited** to zonotope geometry (hybrid zonotope equivalence)
7. **GIoU-style losses** provide gradient signal for non-overlapping cases
