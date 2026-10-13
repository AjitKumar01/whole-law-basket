# Mathematical Theory and Architecture of the Whole-Law Basket Model

This document provides a complete, first-principles mathematical treatment of the **Whole-Law Basket Choice and Generation Architecture** implemented in [`whole_law_basket`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket):
1. **Vectorized Differentiable Exponential-Race Plackett–Luce (PL) Quadrature** ([`race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py)) for scalable within-category unordered subset choice at arbitrary category width $P$.
2. **`CompositeWholeLawEnergy`** ([`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) for real retail scanner panels (ERIM and Dunnhumby), unifying category-frontier dynamic programming (DP), nonparametric global basket-size potentials, size-1 quick-trip stratification, and exact-partition category and product mission mixtures.
3. **`CalibratedMissionLaw`** ([`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) for latent-mission counterfactual price response, combining differentiable race-PL subset choice with non-negative hierarchical cross-price shrinkage.
4. **Exact Comparison Against 11 Multi-Paradigm Baselines** ([`baselines.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/baselines.py)), where every model is normalized over the exact same support of $2^J - 1$ nonempty binary baskets.

---

## 1. Problem Formulation, Support, and Notation

### 1.1 Basket Choice on a Structured Catalogue
Consider a retail assortment organized into $C$ categories, each containing $P$ distinct products, for a total catalogue of $J = C \times P$ products. A shopping basket is represented as a binary indicator matrix

$$y \in \{0, 1\}^{C \times P} \cong \{0, 1\}^J,$$

where $y_{c,p} = 1$ if product $p \in \{1, \dots, P\}$ in category $c \in \{1, \dots, C\}$ is purchased on the trip, and $y_{c,p} = 0$ otherwise. Because checkout records only register trips in which at least one catalogue item is purchased, the sample space is the set of all **nonempty binary baskets**:

$$\mathcal{S}_+ = \{0, 1\}^J \setminus \{\mathbf{0}\}, \qquad |\mathcal{S}_+| = 2^J - 1.$$

For any basket $y \in \mathcal{S}_+$, we define the following structural summaries:

| Symbol | Definition | Interpretation |
| :--- | :--- | :--- |
| $z = (h, s, t, d, D, M)$ | Context tuple | Household $h$, store $s$, week $t$, log-price $d$, display $D$, mailer $M$ |
| $y_c \in \{0, 1\}^P$ | $(y_{c,1}, \dots, y_{c,P})$ | Within-category product selection vector in category $c$ |
| $n_c(y) \in \{0, \dots, P\}$ | $\sum_{p=1}^P y_{c,p}$ | Number of distinct products purchased in category $c$ |
| $x_c(y) \in \{0, 1\}$ | $\mathbb{I}(n_c(y) > 0)$ | Category incidence indicator ($1$ if category $c$ is visited) |
| $N(y) \in \{1, \dots, J\}$ | $\sum_{c=1}^C n_c(y) = \sum_{j=1}^J y_j$ | Total distinct basket size across the catalogue |
| $d_{c,p} \in \mathbb{R}$ | $\log(p_{c,p} / p_{c,p}^{\text{ref}})$ | Log offered price relative to fixed training reference price $p_{c,p}^{\text{ref}}$ |
| $b_j(z) \in \mathbb{R}$ | $\lambda_j + \theta_h^\top \alpha_j + \xi_s^\top \zeta_j + f(t)^\top \mu_j + w_j^D D_j + w_j^M M_j$ | Price-free baseline preference from certified parent model |
| $u_j(z) \in \mathbb{R}$ | $b_j(z) + \tilde{\lambda}_j + H_{h,j} - \beta_j^- \min(d_j, 0) - \beta_j^+ \max(d_j, 0)$ | Net product utility under asymmetric or linear price response |

### 1.2 Why Exact Whole-Law Normalization Over $\mathcal{S}_+$ Is Required
A generative basket simulator must simultaneously support:
1. **Exact held-out log-likelihood evaluation**: $\log P_\vartheta(y \mid z)$ for any observed basket $y \in \mathcal{S}_+$ without Monte Carlo partition variance or pseudolikelihood approximation.
2. **Exact, rejection-free basket generation**: Drawing $y \sim P_\vartheta(\cdot \mid z)$ in polynomial time, even when the unconditional probability of a nonempty basket $\mathbf{P}(y \neq \mathbf{0})$ is small.
3. **Coherent price counterfactuals ("what-if" scenarios)**: Computing shifts in item incidence $\mathbb{E}[y_j \mid z + \Delta d]$, category participation $\mathbb{E}[x_c \mid z + \Delta d]$, and basket size $\mathbb{E}[N(y) \mid z + \Delta d]$ under a single self-consistent probability law satisfying $\sum_{y \in \mathcal{S}_+} P_\vartheta(y \mid z) = 1$.

---

## 2. Theoretical Diagnosis of the Three Prior Bottlenecks

Before deriving our proposed architecture, we formalize the three mathematical bottlenecks that caused earlier models to stall:

### Bottleneck 1: The $O(P \cdot 2^P)$ Subset Enumeration Limit in Plackett–Luce Choice
In a category with $P$ products and item utilities $u_c = (u_{c,1}, \dots, u_{c,P})$, a shopper who draws $n_c = k$ distinct items without replacement according to a Plackett–Luce (PL) process generates an ordered sequence $\pi = (\pi_1, \dots, \pi_k)$ with probability

$$P_{\text{PL}}^{\text{ord}}(\pi \mid u_c) = \prod_{r=1}^k \frac{\exp(u_{c, \pi_r})}{\sum_{j \notin \{\pi_1, \dots, \pi_{r-1}\}} \exp(u_{c, j})}.$$

Because checkout scanner data record only the **unordered subset** $A = \{\pi_1, \dots, \pi_k\} \subset \{1, \dots, P\}$, evaluating the exact unordered subset probability

$$P_{\text{PL}}(A \mid u_c) = \sum_{\pi \in \mathfrak{S}(A)} P_{\text{PL}}^{\text{ord}}(\pi \mid u_c)$$

by subset dynamic programming (`subset_log_probabilities` in `MissionLaw`) requires iterating over all $2^P$ subsets and $P$ transitions per subset, costing $O(P \cdot 2^P)$ operations. At $P = 4$ ($2^4 = 16$), this is trivial; at Dunnhumby full category width $P = 16$ ($2^{16} = 65,536$) or $P = 64$ ($2^{64} \approx 1.84 \times 10^{19}$), subset DP is computationally intractable.

### Bottleneck 2: Cross-Category Price Collinearity in Latent-Mission Worlds
In `MissionLaw`, category participation logits $\eta_{n,m,c}$ for household $h_n$ in mission $m \in \{1, 2\}$ depend on own-category mean log-price $\bar{d}_{n,c} = \frac{1}{P}\sum_{p=1}^P d_{n,c,p}$ and adjacent-category mean log-price $\bar{d}_{n, c-1}$:

$$\eta_{n,m,c} = \kappa_c + H_{h_n, c}^{\text{need}} \pm \frac{1}{2} \delta_c^{\text{need}} - \beta_c^{\text{cat}} \bar{d}_{n,c} + \gamma_c^{\text{cross}} \bar{d}_{n, c-1} - \beta_c^{\text{hinge}} \max(\bar{d}_{n,c}, 0).$$

Because category participation $x_{n,c} \in \{0, 1\}$ is binary, censored by the nonempty-trip normalizer $1 - \prod_{c=1}^C (1 - \sigma(\eta_{n,m,c}))$, and mixed over two latent missions $m \in \{1, 2\}$, the Fisher information for the cross-category price vector $\gamma^{\text{cross}} = (\gamma_1^{\text{cross}}, \dots, \gamma_C^{\text{cross}})$ is poorly conditioned along contrast directions $\sum_c \gamma_c^{\text{cross}} = \text{const}$. Under unconstrained MLE on $20,000$ training baskets (`missions_a`), sampling noise drives one coordinate positive and another negative (`[+0.1169, -0.0717, -0.0038]`), inflating cross-category counterfactual relative MAE to `0.3211` (above the `0.25` threshold).

### Bottleneck 3: The Real-Data Transfer Failure of Unstratified `StructuredEnergy`
The previous `StructuredEnergy` model defines the basket log-probability via a quadratic count energy:

$$\log P_{\text{SE}}(y \mid z) = \sum_{j=1}^J u_j(z) y_j + \sum_{c=1}^C \gamma_c \frac{n_c(y)(n_c(y)-1)}{2} + \gamma_{\text{global}} \frac{N(y)(N(y)-1)}{2} + \sum_{(i,j) \in \mathcal{E}} \theta_{ij} y_i y_j - \log Z_{\text{SE}}(z).$$

On real scanner data (especially Dunnhumby with $J = 64$ products across $C = 16$ categories), `StructuredEnergy` lost to `FastNeuralNADE` on held-out test log-likelihood (`-6.9470` vs `-6.8426`) and suffered a $4.3\times$ worse basket-size Total Variation distance (`size_TV = 0.0816` vs `0.0189`). Three structural misspecifications caused this gap:
1. **The Quadratic Count Trap**: A single scalar $\gamma_{\text{global}} \binom{N}{2}$ cannot simultaneously fit a high mass of single-item quick trips ($N = 1$, which is $\sim 35\%$ of projected baskets), a sharp drop at $N = 2, 3$, and a heavy right tail ($N = 8, \dots, 20$) without inducing severe bimodal or over-dispersed distortion in $P(N)$.
2. **Conflation of Quick-Trip ($N=1$) and Stock-Up ($N \ge 2$) Item Preferences**: Items purchased on single-item fill-in trips (e.g., milk, bananas, bread, convenience items) have systematically different baseline frequencies and cross-price elasticities than items purchased during large multi-category stock-up trips.
3. **Lack of Low-Rank Cross-Category Mission Bundles**: `StructuredEnergy` restricts cross-category interactions to at most $|\mathcal{E}| \le 8$ sparse product edges to keep the DP frontier width $\le 2$. In contrast, `FastNeuralNADE` uses a dense rank-16 hidden prefix across all $J$ products. Real shopping trips exhibit latent shopping missions (e.g., breakfast mission, baking mission, cleaning mission) that couple all $C$ categories simultaneously.

---

## 3. Solution Part I: Differentiable Exponential-Race Plackett–Luce Quadrature

### 3.1 Exponential-Race Representation of Unordered Plackett–Luce Choice
Let $u_c = (u_{c,1}, \dots, u_{c,P}) \in \mathbb{R}^P$ be the product utilities in category $c$, and define the normalized positive rates

$$w_i = \frac{\exp(u_{c,i})}{\sum_{m=1}^P \exp(u_{c,m})}, \qquad \sum_{i=1}^P w_i = 1.$$

By the classic Thurstonian / Gumbel-race construction of the Plackett–Luce model, drawing an ordered sequence without replacement according to rates $(w_1, \dots, w_P)$ is distributionally identical to assigning each product $i \in \{1, \dots, P\}$ an independent exponential arrival time

$$T_i \stackrel{\text{ind}}{\sim} \operatorname{Exponential}(w_i), \qquad f_{T_i}(\tau) = w_i e^{-w_i \tau} \quad (\tau \ge 0),$$

and sorting products in increasing order of their arrival times $T_{\pi_1} < T_{\pi_2} < \dots < T_{\pi_P}$.

Consequently, a nonempty proper subset $A \subset \{1, \dots, P\}$ ($0 < |A| < P$) constitutes the exact set of the first $k = |A|$ chosen items if and only if **every item in $A$ arrives strictly before every item in the complement $A^c = \{1, \dots, P\} \setminus A$**:

$$\max_{i \in A} T_i < \min_{j \in A^c} T_j.$$

Let $T_{\text{out}} = \min_{j \in A^c} T_j$. Because the minimum of independent exponential random variables is itself exponential with rate equal to the sum of the rates,

$$T_{\text{out}} \sim \operatorname{Exponential}(w_{\text{out}}), \qquad w_{\text{out}} = \sum_{j \in A^c} w_j = 1 - \sum_{i \in A} w_i > 0.$$

Conditioning on $T_{\text{out}} = \tau$ and using the conditional independence of $\{T_i\}_{i \in A}$, each $i \in A$ satisfies $\mathbf{P}(T_i < \tau) = 1 - e^{-w_i \tau}$. Integrating over the density $f_{T_{\text{out}}}(\tau) = w_{\text{out}} e^{-w_{\text{out}} \tau}$ yields the exact 1D integral identity:

$$P_{\text{PL}}(A \mid u_c) = \int_0^\infty w_{\text{out}} e^{-w_{\text{out}} \tau} \prod_{i \in A} \left(1 - e^{-w_i \tau}\right) d\tau.$$

### 3.2 Unit-Rate Rescaling and Rational Gauss–Legendre Quadrature
If we apply quadrature directly in $\tau \in [0, \infty)$, the characteristic decay scale $1 / w_{\text{out}}$ varies across subsets and category sizes $P$ by orders of magnitude (e.g., as $P \to 64$ or as $\sum_{i \in A} w_i \to 1$). To make the integrand scale-invariant, we first perform the **unit-rate change of variables**

$$s = w_{\text{out}} \tau \quad \Longleftrightarrow \quad \tau = \frac{s}{w_{\text{out}}}, \qquad d\tau = \frac{ds}{w_{\text{out}}},$$

which transforms the integral into the canonical unit-exponential form:

$$P_{\text{PL}}(A \mid u_c) = \int_0^\infty e^{-s} \prod_{i \in A} \left(1 - \exp\!\left(-\frac{w_i}{w_{\text{out}}} s\right)\right) ds.$$

Next, we map the half-line $s \in (0, \infty)$ onto the unit interval $t \in (0, 1)$ via the rational square transformation

$$s(t) = \left(\frac{t}{1 - t}\right)^2, \qquad \frac{ds}{dt} = \frac{2t}{(1 - t)^3}.$$

Why $s(t) = (t/(1-t))^2$?
1. Near $t \to 0^+$, $s(t) = O(t^2)$ and $ds/dt = O(t)$, which clusters quadrature nodes densely near $s = 0$ where the product $\prod_{i \in A} (1 - e^{-(w_i / w_{\text{out}}) s}) \sim s^{|A|}$ has high-order curvature for multiple selections $|A| \ge 2$.
2. Near $t \to 1^-$, $e^{-s(t)} = \exp(-(t/(1-t))^2)$ decays super-exponentially, making all derivatives vanish at $t = 1$ and guaranteeing spectral (exponential) convergence of Gauss–Legendre quadrature.

Let $(\xi_q, \varpi_q)_{q=1}^M$ be the standard $M$-point Gauss–Legendre nodes and weights on $[-1, 1]$, and define the shifted nodes and log-Jacobian weights on $(0, 1)$:

$$t_q = \frac{\xi_q + 1}{2}, \qquad \omega_q = \frac{\varpi_q}{2}, \qquad s_q = \left(\frac{t_q}{1 - t_q}\right)^2, \qquad \ell_q = \log\!\left(\omega_q \frac{2 t_q}{(1 - t_q)^3}\right) - s_q.$$

Then for any observed binary selection mask $y_c \in \{0, 1\}^P$ with $0 < n_c < P$, the log-probability is given by a single vectorized `logsumexp` over $q \in \{1, \dots, M\}$ (`race_observed_log_probability` in [`race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py)):

$$\boxed{
\log P_{\text{PL}}(y_c \mid u_c) = \operatorname{logsumexp}_{q=1}^M \left[ \ell_q + \sum_{p=1}^P y_{c,p} \log\!\left(-\operatorname{expm1}\!\left(-\frac{w_p}{w_{\text{out}}} s_q\right)\right) \right],
}$$

with boundary conditions $\log P_{\text{PL}}(\mathbf{0} \mid u_c) = \log P_{\text{PL}}(\mathbf{1} \mid u_c) = 0$.

### 3.3 Complexity, Shift-Invariance, and Machine-Precision Verification
* **Complexity**: Computing $\log P_{\text{PL}}(y_c \mid u_c)$ and its exact PyTorch reverse-mode gradient $\nabla_{u_c} \log P_{\text{PL}}(y_c \mid u_c)$ costs $O(P \cdot M)$ time and memory (with $M = 64$), replacing exponential $O(P \cdot 2^P)$ enumeration with linear $O(P)$ scaling.
* **Exact Shift-Invariance**: Because $w_p / w_{\text{out}} = \exp(u_{c,p}) / \sum_{j: y_{c,j}=0} \exp(u_{c,j})$ is invariant to adding any constant $c \mathbf{1}$ to $u_c$, the quadrature formula is identically shift-invariant, satisfying $\sum_{p=1}^P \frac{\partial \log P_{\text{PL}}(y_c \mid u_c)}{\partial u_{c,p}} = 0$ to floating-point machine epsilon (`1.11e-16`).
* **Empirical Verification ([`benchmark_summary.json`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/benchmark_summary.json))**:
  - At $P = 4$ across all $2^4 = 16$ subsets: maximum log-probability discrepancy vs. exact $2^P$ DP is **`5.83e-13`**; maximum autograd gradient discrepancy is **`2.39e-13`**.
  - At $P = 16$ ($1,024$-product Dunnhumby category width) and $P = 64$: maximum log-probability error against exact combinatorial truth $-\log\binom{P}{3}$ is **`8.01e-13`** ($P=16$) and **`6.75e-13`** ($P=64$), executing forward + backward passes in **`0.007–0.010s`**.

---

## 4. Solution Part II: `CompositeWholeLawEnergy` for Real Scanner Panels

To resolve the real-data transfer bottleneck on ERIM ($C=7, P=4, J=28$) and Dunnhumby ($C=16, P=4, J=64$), `CompositeWholeLawEnergy` ([`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) constructs an **exact-normalized whole-basket mixture** over $\mathcal{S}_+ = \{0, 1\}^J \setminus \{\mathbf{0}\}$ combining three complementary generative mechanisms:
1. **Component $0$ ($P_0$)**: A size-stratified category-frontier DP energy law with a nonparametric global basket-size potential $g(N)$, household-specific count slope $s_h(N-1)$, and a dedicated size-1 quick-trip utility stratum $W_1(z)$.
2. **Components $1, \dots, K_c$ ($P_k^{\text{cat}}$)**: Pre-trip category mission laws that couple category incidence across all $C$ categories while preserving within-category count curvature and intra-category product links.
3. **Components $1, \dots, K_p$ ($P_m^{\text{prod}}$)**: Cross-category product bundle mission laws initialized from the top singular vectors of training product co-occurrence.

### 4.1 Inherited Category-Frontier Dynamic Program (`StructuredEnergy`)
Let $\mathcal{E} \subset \{(i, j) : 1 \le i < j \le J\}$ be the sparse set of product interaction edges selected by the convex training-only conditional logistic screen (`screen.json`), with learned interaction weights $\theta_e \in \mathbb{R}$ for $e \in \mathcal{E}$ and within-category count curvature $\gamma_c \in \mathbb{R}$. Define the un-tilted base energy of basket $y \in \mathcal{S}_+$ (with the quadratic global count term removed):

$$E_{\text{base}}(y \mid z) = \sum_{j=1}^J u_j(z) y_j + \sum_{c=1}^C \gamma_c \frac{n_c(y)(n_c(y) - 1)}{2} + \sum_{(i, j) \in \mathcal{E}} \theta_{ij} y_i y_j.$$

By sweeping through categories $c = 1, \dots, C$ and maintaining only the active boundary endpoints $B_c = \{i \le c P : \exists (i, j) \in \mathcal{E} \text{ with } j > c P\}$ and the running basket size $N \in \{0, 1, \dots, c P\}$, the category-frontier dynamic program (`StructuredEnergy.partition` in [`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) computes the exact log-sum of $\exp(E_{\text{base}}(y \mid z))$ for **every fixed total basket size $N \in \{1, \dots, J\}$**:

$$W_{\text{base}}(N \mid z) = \log \sum_{y \in \mathcal{S}_+:\, N(y) = N} \exp\!\big(E_{\text{base}}(y \mid z)\big), \qquad N = 1, \dots, J.$$

Crucially, because $u_j(z)$, $\gamma_c$, and $\theta_{ij}$ are inherited from the converged `StructuredEnergy` checkpoint, the entire vector $(W_{\text{base}}(1 \mid z), \dots, W_{\text{base}}(J \mid z)) \in \mathbb{R}^J$ is **precomputed once** per split in `precompute_split(ctx, y)`.

### 4.2 Sub-Law 1 ($P_0$): Nonparametric Count Potential, Shopper Slope & Size-1 Quick-Trip Stratum
In `CompositeWholeLawEnergy`, Component $0$ replaces the rigid quadratic penalty $\gamma_{\text{global}}\binom{N}{2}$ with three exact structural upgrades:

1. **Dedicated Size-1 Quick-Trip Utility Stratum ($N = 1$)**:
   For single-item trips ($N(y) = 1$, where $y = e_j$ for a single product $j \in \{1, \dots, J\}$), product $j$ has utility

   $$u_j^{(1)}(z) = u_j(z) + \delta_j^{(1)} + H_{h, j}^{(1)} + \big[(d_z R_1) L_1^\top - (L_1 \odot R_1)\mathbf{1} \odot d_z\big]_j,$$

   where $\delta^{(1)} \in \mathbb{R}^J$ is the size-1 product intercept shift, $H^{(1)} \in \mathbb{R}^{|\mathcal{H}| \times J}$ is the household size-1 preference table (frozen to zero on Dunnhumby $J=64$ to prevent $159,680$-parameter overfitting, and active on ERIM $J=28$), and $L_1, R_1 \in \mathbb{R}^{J \times 8}$ parameterize a zero-diagonal rank-8 cross-price bridge on quick trips. The exact log-partition of all $\binom{J}{1} = J$ single-item baskets is simply:

   $$W_1(z) = \log \sum_{j=1}^J \exp\!\big(u_j^{(1)}(z)\big) = \operatorname{logsumexp}_{j=1}^J u_j^{(1)}(z).$$

2. **Nonparametric Global Basket-Size Potential $g(N)$**:
   For multi-item baskets ($N \ge 2$), we introduce a nonparametric size potential $g: \{1, \dots, J\} \to \mathbb{R}$ anchored at $g(1) = 0$:

   $$g(N) = \begin{cases} 0, & N = 1, \\ \tilde{g}_N, & 2 \le N \le N_{\max} \quad (N_{\max} = \min(J, 17)), \\ \tilde{g}_{N_{\max}} - \operatorname{softplus}(\beta_{\text{tail}})(N - N_{\max}), & N > N_{\max}. \end{cases}$$

   This allows the model to match the exact empirical shape of the basket-size distribution at every count $N = 1, 2, \dots, 17$ while guaranteeing a monotone exponential tail for $N > 17$.

3. **Household-Specific Basket-Size Slope $s_h$**:
   Each household $h \in \mathcal{H}$ has a scalar trip-size propensity $s_h \in \mathbb{R}$ multiplying $(N - 1)$, capturing persistent heterogeneity between small-basket fill-in shoppers and large-basket stock-up shoppers.

Combining these terms yields the exact size-$N$ log-weights and total partition function $Z_0(z)$ for Component $0$:

$$W_0(N \mid z) = \begin{cases} W_1(z), & N = 1, \\ W_{\text{base}}(N \mid z) + g(N) + s_h (N - 1), & N \in \{2, \dots, J\}, \end{cases}$$

$$Z_0(z) = \operatorname{logsumexp}_{N=1}^J W_0(N \mid z),$$

and the exact normalized log-probability of any nonempty basket $y \in \mathcal{S}_+$ under Component $0$:

$$\log P_0(y \mid z) = \begin{cases} \sum_{j=1}^J u_j^{(1)}(z) y_j - Z_0(z), & N(y) = 1, \\ E_{\text{base}}(y \mid z) + g(N(y)) + s_h (N(y) - 1) - Z_0(z), & N(y) \ge 2. \end{cases}$$

**Verification of Exact Normalization**: Summing $\exp(\log P_0(y \mid z))$ over all $y \in \mathcal{S}_+$ partitions $\mathcal{S}_+$ by size $N \in \{1, \dots, J\}$:
- For $N = 1$: $\sum_{y:\, N(y)=1} \exp(\sum_j u_j^{(1)}(z) y_j) = \sum_{j=1}^J \exp(u_j^{(1)}(z)) = \exp(W_1(z)) = \exp(W_0(1 \mid z))$.
- For $N \ge 2$: $\sum_{y:\, N(y)=N} \exp(E_{\text{base}}(y \mid z) + g(N) + s_h(N-1)) = \exp(W_{\text{base}}(N \mid z) + g(N) + s_h(N-1)) = \exp(W_0(N \mid z))$.
Summing over $N = 1, \dots, J$ gives $\sum_{N=1}^J \exp(W_0(N \mid z)) = \exp(Z_0(z))$, so $\sum_{y \in \mathcal{S}_+} P_0(y \mid z) = 1$ identically.

### 4.3 Sub-Law 2 ($P_k^{\text{cat}}$): Exact-Partition Pre-Trip Category Mission Mixture
Let $\mathcal{S}_c^+ = \{0, 1\}^P \setminus \{\mathbf{0}\}$ denote the $2^P - 1$ nonempty product subsets within category $c \in \{1, \dots, C\}$, and let $\mathcal{E}_{\text{within}, c} = \{(i, j) \in \mathcal{E} : \lfloor i/P \rfloor = \lfloor j/P \rfloor = c\}$ be the within-category edges in category $c$. For any basket $y \in \mathcal{S}_+$, define its within-category base energy:

$$E_{\text{within}}(y \mid z) = \sum_{c=1}^C \left[ \sum_{p=1}^P u_{c,p}(z) y_{c,p} + \gamma_c \frac{n_c(y)(n_c(y) - 1)}{2} + \sum_{(i, j) \in \mathcal{E}_{\text{within}, c}} \theta_{ij} y_i y_j \right].$$

Summing $\exp(\cdots)$ over all $2^P - 1$ nonempty subsets $s \in \mathcal{S}_c^+$ in category $c$ gives the exact within-category inclusive weight:

$$\log W_c^{\text{cat}}(z) = \log \sum_{s \in \mathcal{S}_c^+} \exp\!\left( u_c(z)^\top s + \gamma_c \frac{|s|(|s|-1)}{2} + \sum_{(i,j) \in \mathcal{E}_{\text{within}, c}} \theta_{ij} s_{i \bmod P} s_{j \bmod P} \right).$$

We introduce a contextual category participation shift

$$\Delta_c(z) = \kappa_c^{\text{shift}} + H_{h, c}^{\text{cat}} + \big(d_z \Phi^{\text{cat}}\big)_c$$

and $K_c$ category mission prototype vectors $M_k^{\text{cat}} \in \mathbb{R}^C$ ($k = 1, \dots, K_c$, initialized from 4 global size-tier intercepts plus the top $\min(6, C)$ positive and negative singular vectors of centered training category co-occurrence $x - \bar{x}$).

Under category mission $k \in \{1, \dots, K_c\}$, each category $c$ has total nonempty log-weight

$$\eta_{k,c}^{\text{cat}}(z) = \log W_c^{\text{cat}}(z) + \Delta_c(z) + M_{k,c}^{\text{cat}}.$$

Because category subsets factorize across $c = 1, \dots, C$ conditional on mission $k$, summing over all $2^J - 1$ nonempty baskets $y \in \mathcal{S}_+$ yields the closed-form log-partition:

$$\log Z_k^{\text{cat}}(z) = \log\!\left( \prod_{c=1}^C \Big(1 + \exp\big(\eta_{k,c}^{\text{cat}}(z)\big)\Big) - 1 \right) = A_k^{\text{cat}}(z) + \log\!\Big(-\operatorname{expm1}\big(-A_k^{\text{cat}}(z)\big)\Big),$$

where $A_k^{\text{cat}}(z) = \sum_{c=1}^C \operatorname{softplus}\big(\eta_{k,c}^{\text{cat}}(z)\big)$. Thus the exact log-probability of $y \in \mathcal{S}_+$ under category mission $k$ is:

$$\log P_k^{\text{cat}}(y \mid z) = E_{\text{within}}(y \mid z) + \sum_{c=1}^C x_c(y) \big(\Delta_c(z) + M_{k,c}^{\text{cat}}\big) - \log Z_k^{\text{cat}}(z).$$

### 4.4 Sub-Law 3 ($P_m^{\text{prod}}$): Exact-Partition Product Bundle Mission Mixture
To capture cross-category product-specific bundles (e.g., specific brand or recipe pairings across categories), we extract the top 8 singular vectors $V_1, \dots, V_8 \in \mathbb{R}^J$ of the centered training product matrix $y - \bar{y}$ and construct $K_p = 16$ fixed signed product mission prototypes $M_m^{\text{prod}} \in \mathbb{R}^J$ ($m = 1, \dots, K_p$), augmented by a trainable category-level modulation matrix $M_{m, c}^{\text{prod-cat}} \in \mathbb{R}^{K_p \times C}$.

For each product mission $m \in \{1, \dots, K_p\}$ and category $c \in \{1, \dots, C\}$, the within-category nonempty subset sum with shifted product utilities $u_c(z) + M_{m, c}^{\text{prod}}$ is precomputed in `precompute_split`:

$$\log W_{m, c}^{\text{prod}}(z) = \log \sum_{s \in \mathcal{S}_c^+} \exp\!\left( \big(u_c(z) + M_{m,c}^{\text{prod}}\big)^\top s + \gamma_c \frac{|s|(|s|-1)}{2} + \sum_{(i,j) \in \mathcal{E}_{\text{within}, c}} \theta_{ij} s_{i \bmod P} s_{j \bmod P} \right).$$

Adding the category shift $\Delta_c(z) + M_{m, c}^{\text{prod-cat}}$ gives the effective category log-weight

$$\eta_{m, c}^{\text{prod}}(z) = \log W_{m, c}^{\text{prod}}(z) + \Delta_c(z) + M_{m, c}^{\text{prod-cat}},$$

with exact nonempty partition across all $2^J - 1$ baskets:

$$\log Z_m^{\text{prod}}(z) = A_m^{\text{prod}}(z) + \log\!\Big(-\operatorname{expm1}\big(-A_m^{\text{prod}}(z)\big)\Big), \qquad A_m^{\text{prod}}(z) = \sum_{c=1}^C \operatorname{softplus}\big(\eta_{m, c}^{\text{prod}}(z)\big),$$

and exact normalized log-probability:

$$\log P_m^{\text{prod}}(y \mid z) = E_{\text{within}}(y \mid z) + \sum_{j=1}^J M_{m, j}^{\text{prod}} y_j + \sum_{c=1}^C x_c(y) \big(\Delta_c(z) + M_{m, c}^{\text{prod-cat}}\big) - \log Z_m^{\text{prod}}(z).$$

### 4.5 Complete `CompositeWholeLawEnergy` Mixture Law
Let $K_{\text{tot}} = 1 + K_c + K_p$ be the total number of mixture components. The context-dependent mixture weights $\pi(z) = (\pi_0(z), \pi_1^{\text{cat}}(z), \dots, \pi_{K_p}^{\text{prod}}(z)) \in \Delta^{K_{\text{tot}} - 1}$ are given by

$$\log \pi(z) = \operatorname{log\_softmax}\!\Big(\alpha^{\text{mix}} + H_h^{\text{mix}} + d_z \Phi^{\text{mix}}\Big).$$

The complete basket probability law is the log-sum-exp over all $K_{\text{tot}}$ exact-normalized components:

$$\boxed{
\log P_{\text{comp}}(y \mid z) = \operatorname{logsumexp}\!\left( \log \pi_0(z) + \log P_0(y \mid z), \; \big\{\log \pi_k^{\text{cat}}(z) + \log P_k^{\text{cat}}(y \mid z)\big\}_{k=1}^{K_c}, \; \big\{\log \pi_m^{\text{prod}}(z) + \log P_m^{\text{prod}}(y \mid z)\big\}_{m=1}^{K_p} \right).
}$$

Because $P_0(\cdot \mid z)$, every $P_k^{\text{cat}}(\cdot \mid z)$, and every $P_m^{\text{prod}}(\cdot \mid z)$ sums to $1$ over $\mathcal{S}_+$, and $\sum_{r=1}^{K_{\text{tot}}} \pi_r(z) = 1$ with $\pi_r(z) > 0$, the composite law satisfies $\sum_{y \in \mathcal{S}_+} P_{\text{comp}}(y \mid z) = 1$ identically for all contexts $z$ and all parameter values.

---

## 5. Exact Rejection-Free Ancestral & Backward-DP Sampling

A central advantage of `CompositeWholeLawEnergy` is that its exact sampler (`CompositeWholeLawEnergy.sample` in [`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) requires **zero Markov chain Monte Carlo (MCMC) burn-in and zero rejection sampling**. Given a batch of contexts $z$, sampling proceeds in two stages:

### Stage 1: Mixture Component Selection
Draw the active component index $r \sim \operatorname{Categorical}(\pi(z)) \in \{0, 1, \dots, K_c + K_p\}$.

### Stage 2A: If $r = 0$ (Size-Stratified DP Energy Component)
1. **Draw Total Basket Size $N^*$**:
   Sample $N^* \in \{1, \dots, J\}$ from the exact marginal basket-size distribution:
   $$P_0(N(y) = N \mid z) = \frac{\exp(W_0(N \mid z))}{\sum_{n=1}^J \exp(W_0(n \mid z))}.$$
2. **If $N^* = 1$ (Single-Item Quick Trip)**:
   Draw the single purchased product $j^* \sim \operatorname{Categorical}\!\big(\operatorname{softmax}(u^{(1)}(z))\big)$ and set $y = e_{j^*}$.
3. **If $N^* \ge 2$ (Multi-Item Stock-Up Trip via Backward Frontier DP)**:
   - Select the terminal DP state $s_C$ at layer $C$ conditional on $N(s_C) = N^*$ with probability proportional to $\exp(W_{\text{final}}(s_C \mid z))$.
   - Sweep backwards through categories $c = C, C-1, \dots, 1$:
     - Given destination state $s_c$, sample the incoming transition $t_c \in \operatorname{Incoming}(s_c)$ with probability proportional to the forward path weight $\exp(V_c(t_c \mid z))$.
     - Set the interface product indicators $y_{c, \text{interface}}$ directly from the transition bitmask $\operatorname{mask}(t_c)$.
     - Given the required number of free products $n_{\text{free}} = \operatorname{extra}(t_c)$ in category $c$, sample an unordered subset of size $n_{\text{free}}$ from the free products $F_c$ using exact backward Elementary Symmetric Polynomial (ESP) suffix ratios:
       $$\mathbf{P}\!\left(y_{c, f_k} = 1 \;\middle|\; \text{remaining count} = m\right) = \frac{\exp(u_{c, f_k}(z)) \, E_{m-1}(u_{c, f_{>k}}(z))}{E_m(u_{c, f_{\ge k}}(z))}.$$
     - Update the state $s_{c-1} = \operatorname{old}(t_c)$.

### Stage 2B: If $r \ge 1$ (Category or Product Mission Component)
1. **Exact Nonempty Category Pattern Sampling**:
   Let $\eta_c = \eta_{r, c}(z)$ be the category log-weight for $c = 1, \dots, C$, so the unconditional category presence probability is $\sigma(\eta_c) = \frac{\exp(\eta_c)}{1 + \exp(\eta_c)}$. To condition on at least one active category ($x \neq \mathbf{0}$) without rejection, compute the backward log-probability of eventual nonemptiness on the all-zero prefix:
   $$S_c = \log\!\Big(1 - \prod_{d=c}^C \big(1 - \sigma(\eta_d)\big)\Big) = \log\Big(\sigma(\eta_c) + \big(1 - \sigma(\eta_c)\big) e^{S_{c+1}}\Big), \qquad S_{C+1} = -\infty.$$
   Sweep forward $c = 1, \dots, C$, maintaining an indicator $\text{active} \in \{0, 1\}$ of whether any category $d < c$ has been selected:
   $$\mathbf{P}(x_c = 1 \mid x_{<c}, x \neq \mathbf{0}) = \begin{cases} \exp(\log \sigma(\eta_c) - S_c), & \text{if } \text{active} = 0, \\ \sigma(\eta_c), & \text{if } \text{active} = 1. \end{cases}$$
2. **Exact Within-Category Subset Sampling**:
   For each active category $c$ ($x_c = 1$), draw one of the $2^P - 1$ nonempty product subsets $s_c \in \mathcal{S}_c^+$ from its exact categorical distribution:
   $$\mathbf{P}(y_c = s \mid x_c = 1, r, z) = \operatorname{softmax}_{s \in \mathcal{S}_c^+}\!\left( u_{r,c}^{\text{eff}}(z)^\top s + \gamma_c \frac{|s|(|s|-1)}{2} + \sum_{(i,j) \in \mathcal{E}_{\text{within}, c}} \theta_{ij} s_{i \bmod P} s_{j \bmod P} \right).$$

---

## 6. Solution Part III: Hierarchical Cross-Price Pooling in `CalibratedMissionLaw`

### 6.1 Variance-Bias Analysis of Cross-Category Price Slopes
In synthetic mission benchmarks (`missions_a`, $C = 3, P = 4$), category participation logits follow

$$\eta_{n,m,c} = \kappa_c + H_{h_n, c}^{\text{need}} + \nu_m \delta_c^{\text{need}} - \beta_c^{\text{cat}} \bar{d}_{n,c} + \gamma_c^{\text{cross}} \bar{d}_{n, c-1} - \beta_c^{\text{hinge}} \max(\bar{d}_{n,c}, 0), \qquad \nu_m \in \{+0.5, -0.5\}.$$

Let $\mathcal{I}(\gamma^{\text{cross}})$ denote the observed Fisher information submatrix for $\gamma^{\text{cross}} \in \mathbb{R}^C$. Because the nonempty trip normalizer $\log(1 - \prod_{c=1}^C (1 - \sigma(\eta_{n,m,c})))$ couples all $C$ categories negatively on low-count trips, the off-diagonal entries of $\mathcal{I}^{-1}(\gamma^{\text{cross}})$ are negative, meaning that the average cross-price effect $\bar{\gamma} = \frac{1}{C} \sum_{c=1}^C \gamma_c^{\text{cross}}$ is estimated with much higher precision than the individual category deviations $\gamma_c^{\text{cross}} - \bar{\gamma}$.

In unconstrained MLE (`uncalibrated_mission_law`), $\hat{\gamma}^{\text{MLE}} = (0.1169, -0.0717, -0.0038)$ exhibits severe sign-flipping across categories, even though the true data-generating process has non-negative cross-category substitution $\gamma_c^{\text{cross}} \ge 0$.

### 6.2 Non-Negativity Constraint and Validation-Selected Empirical-Bayes Pooling
In `CalibratedMissionLaw` ([`verify_and_missions.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/artifacts/research_whole_law_solution_20261014/verify_and_missions.py)), we project the raw MLE onto the economically valid positive orthant $\hat{\gamma}_c^+ = \max(\hat{\gamma}_c^{\text{MLE}}, 0)$ and apply hierarchical shrinkage toward a pooled cross-category elasticity $\bar{\gamma} \ge 0$:

$$\hat{\gamma}_c^{\text{cal}}(\bar{\gamma}) = (1 - \alpha) \max(\hat{\gamma}_c^{\text{MLE}}, 0) + \alpha \bar{\gamma}, \qquad \alpha = 0.75,$$

where $\bar{\gamma} \in [0, 0.35]$ is selected solely by maximizing held-out **validation** log-likelihood (never touching test baskets or oracle counterfactuals):

$$\bar{\gamma}^* = \arg\max_{\bar{\gamma} \in [0, 0.35]} \frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{(z_i, y_i) \in \mathcal{D}_{\text{val}}} \log P_{\hat{\gamma}^{\text{cal}}(\bar{\gamma})}(y_i \mid z_i).$$

On `missions_a` (`seed=941000`), validation log-likelihood selects $\hat{\gamma}^{\text{cal}} = (0.1492, 0.1200, 0.1200)$, which:
* Improves held-out test log-likelihood from `-5.23751` to **`-5.23719`** (paired gain `+0.05148` over `FastNeuralNADE`, 95% CI `[+0.0440, +0.0591]`).
* Cuts **cross-category price response relative MAE from `0.3211` to `0.1768`** (well below the $\le 0.25$ target, improving the counterfactual regression slope from `0.6184` to **`0.8756`**).
* Preserves all 9 single/increase/paired counterfactual gate checks (`true`) and achieves `6/6` optimal promotion plans (`0.0` regret).

---

## 7. Mathematical Specification of the 11 Benchmark Baselines

To rigorously test `CompositeWholeLawEnergy` and `CalibratedMissionLaw`, [`whole_law_basket/baselines.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/baselines.py) implements 11 baseline models spanning every major paradigm in econometric choice modeling, statistical physics / graphical models, matrix factorization, variational autoencoders, and deep autoregressive networks. Every baseline is normalized over the exact nonempty support $\mathcal{S}_+ = \{0, 1\}^J \setminus \{\mathbf{0}\}$:

### 7.1 Econometric & Ablation Baselines
1. **`PooledPopularity` (`popularity`)**:
   Item utilities depend only on global product intercepts $u_j(z) = \lambda_j$, conditioned on $y \neq \mathbf{0}$:
   $$\log P_{\text{pop}}(y \mid z) = \sum_{j=1}^J \lambda_j y_j - A(\lambda) - \log\!\big(1 - e^{-A(\lambda)}\big), \qquad A(\lambda) = \sum_{j=1}^J \operatorname{softplus}(\lambda_j).$$
2. **`NoPriceIndependent` (`no_price`)**:
   Includes full household, store, calendar, display, and mailer preferences $u_j(z) = b_j(z) + \tilde{\lambda}_j + H_{h,j}$, with price sensitivity frozen at $\beta_j = 0$:
   $$\log P_{\text{noprice}}(y \mid z) = \sum_{j=1}^J u_j(z) y_j - A(u(z)) - \log\!\big(1 - e^{-A(u(z))}\big).$$
3. **`PersonalizedIndependent` (`independent_parent` / Multivariate Logit Parent)**:
   Adds product-specific price slopes $u_j(z) = b_j(z) + \tilde{\lambda}_j + H_{h,j} - \beta_j d_j$ ($\beta_j \ge 0$), normalized over $\mathcal{S}_+$.

### 7.2 Elementary Symmetric Polynomial (ESP) & Hierarchical Baselines
4. **`SizeAwareESP` (`size_aware_esp`)**:
   Decomposes basket choice into a personalized total basket-size distribution $P(N \mid z) = \operatorname{softmax}_N(\lambda_N^{\text{size}} + (\eta_h + \beta_{\text{size}} \bar{d}_z) N)$ and an Elementary Symmetric Polynomial (ESP) conditional subset draw of size $N$:
   $$\log P_{\text{size-ESP}}(y \mid z) = \log P\big(N(y) \mid z\big) + \sum_{j=1}^J u_j(z) y_j - \log E_{N(y)}\!\big(u(z)\big),$$
   where $E_N(u) = \sum_{T \subset \{1,\dots,J\}:\, |T|=N} \exp(\sum_{j \in T} u_j)$ is computed in $O(J \cdot N_{\max})$ via the log-space DP recurrence $E_n(u_{1:k}) = E_n(u_{1:k-1}) + e^{u_k} E_{n-1}(u_{1:k-1})$.
5. **`IndependentCategoriesESP` (`independent_categories_esp`)**:
   Models within-category counts $n_c \in \{1, \dots, P\}$ via quadratic curvature $\rho_c \binom{n_c}{2} - \nu_c(n_c - 1)$ and couples category participation via inclusive value $\text{IV}_c(z) = \operatorname{logsumexp}_{n=1}^P (\log E_n(u_c(z)) - \rho_c \binom{n}{2} + \nu_c(n-1))$:
   $$\eta_c(z) = \kappa_c + H_{h,c}^{\text{cat}} + \operatorname{softplus}(\tau_c) \text{IV}_c(z),$$
   $$\log P_{\text{cat-ESP}}(y \mid z) = \sum_{c=1}^C \Big[ x_c \log \sigma(\eta_c) + (1 - x_c)\log \sigma(-\eta_c) \Big] - \log\!\Big(1 - \prod_{c=1}^C \sigma(-\eta_c)\Big) + \sum_{c:\, x_c=1} \left[ \log P(n_c \mid x_c=1, z) + \sum_{p=1}^P u_{c,p} y_{c,p} - \log E_{n_c}(u_c) \right].$$
6. **`LatentMissionESP` (`mission_esp`)**:
   A $K=2$ household-weighted latent mixture of `IndependentCategoriesESP` laws:
   $$\log P_{\text{mission-ESP}}(y \mid z) = \operatorname{logsumexp}_{m=1}^K \Big[ \log \pi_{h, m} + \log P_{\text{cat-ESP}}^{(m)}(y \mid z) \Big].$$

### 7.3 Autoregressive, Sequential Factor (SHOPPER), and Variational Baselines
7. **`PairwiseBinaryAR` (`pairwise_ar`)**:
   Autoregressive directed graphical model with strictly lower-triangular pairwise product interaction matrix $W_{\text{tril}} = \operatorname{tril}(W, -1) \in \mathbb{R}^{J \times J}$ and rank-4 cross-price matrix $L_{\text{cp}} R_{\text{cp}}^\top$:
   $$\eta_j(y_{<j}, z) = u_j^{\text{unary}}(z) + \sum_{i < j} W_{j,i} y_i, \qquad \log P_{\text{AR}}(y \mid z) = \sum_{j=1}^J \Big[ y_j \log \sigma(\eta_j) + (1 - y_j) \log \sigma(-\eta_j) \Big] - \log\!\Big(1 - \prod_{j=1}^J \sigma\big(-u_j^{\text{unary}}(z)\big)\Big).$$
8. **`ShopperFactorAR` (`shopper_factor_ar`, inspired by SHOPPER, Ruiz, Athey & Blei 2020)**:
   Parameterizes sequential next-item conditional utilities via rank-$R$ user-item preference factors $\theta_h^\top \alpha_j$, rank-$R$ basket-history interaction factors $\sum_{i < j} y_i (\alpha_j^\top \rho_i)$, and a running basket-size slope $\psi_j \sum_{i < j} y_i$, conditioned exactly on $y \neq \mathbf{0}$:
   $$\eta_j^{\text{SHOPPER}}(y_{<j}, z) = b_j(z) + \lambda_j + \theta_h^\top \alpha_j - \beta_j d_j + \sum_{i < j} (\alpha_j^\top \rho_i) y_i + \psi_j \sum_{i < j} y_i.$$
9. **`LogisticNormalFactorMixture` (`logistic_normal_vae`, adapted from Liang et al. 2018 Multi-VAE / Correlated Basket Factor Model)**:
   Models correlated item co-occurrence via a latent Gaussian trip factor $z_m \in \mathbb{R}^K$ ($m = 1, \dots, M$ quasi-Monte Carlo nodes with $z_1 = \mathbf{0}$) and item factor loadings $V \in \mathbb{R}^{J \times K}$:
   $$u_{m, j}(z) = u_j(z) + (V z_m)_j, \qquad \log P_{\text{VAE}}(y \mid z) = \operatorname{logsumexp}_{m=1}^M \left[ \log w_{h, m} + \sum_{j=1}^J u_{m, j}(z) y_j - A(u_m(z)) - \log\!\big(1 - e^{-A(u_m(z))}\big) \right].$$
10. **`FastNeuralNADE` (`neural_nade`)**:
    Conditional Neural Autoregressive Distribution Estimator with hidden width $H = 16$, rank-8 cross-price bridge, and price-modulated hidden state:
    $$h_j(y_{<j}, d_z) = \sigma\!\left( Q^\top d_z + \sum_{i < j} E_i y_i \right) \in \mathbb{R}^{16}, \qquad \eta_j^{\text{NADE}}(y_{<j}, z) = u_j^{\text{unary}}(z) + O_j^\top h_j(y_{<j}, d_z),$$
    normalized exactly by subtracting $\log(1 - \prod_{j=1}^J \sigma(-\eta_j^{\text{NADE}}(\mathbf{0}, z)))$.
11. **`StructuredEnergy` (`structured_energy`)**:
    The unstratified category-frontier DP energy model with quadratic global count penalty $\gamma_{\text{global}} \binom{N}{2}$ and sparse screened links $\mathcal{E}$.

### 7.4 Architectural Comparison Across Model Families

| Model Family | Exact Partition on $\mathcal{S}_+$ | Arbitrary Size-$N$ Distribution $P(N)$ | Size-1 Quick-Trip Stratification | Cross-Category Mission Coupling | Within-Category Count Curvature | Order-Invariant Joint Law |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CompositeWholeLawEnergy` (Proposed)** | **Yes** | **Yes (Nonparametric $g(N)$)** | **Yes ($W_1$ Stratum)** | **Yes ($K_c + K_p$ Missions)** | **Yes ($\gamma_c + \mathcal{E}_c$)** | **Yes** |
| **`CalibratedMissionLaw` (Proposed)** | **Yes** | **Yes (Mission + Count)** | **Implicit via Count** | **Yes ($K=2$ + Pooled Cross-Price)** | **Yes (Nonparametric Count + Race PL)** | **Yes** |
| `StructuredEnergy` | Yes | No (Quadratic $\binom{N}{2}$ only) | No | Sparse edges ($|\mathcal{E}| \le 8$) | Yes | Yes |
| `FastNeuralNADE` | Yes | Implicit via AR chain | No | Dense $H=16$ prefix | Implicit via AR chain | **No** (Order-dependent) |
| `ShopperFactorAR` / `PairwiseBinaryAR` | Yes | Linear prefix slope only | No | Low-rank / tril prefix | Linear prefix only | **No** (Order-dependent) |
| `LogisticNormalFactorMixture` (VAE) | Yes | Implicit via factor scale | No | Rank-$K$ Gaussian factors | No (Cond. independent) | Yes |
| `LatentMissionESP` / `CategoryESP` | Yes | Sum of category counts | No | $K=2$ / Independent | Yes ($\rho_c \binom{n_c}{2}$) | Yes |
| `SizeAwareESP` | Yes | Yes ($\lambda_N^{\text{size}}$) | Implicit at $N=1$ | No (Global ESP only) | No | Yes |
| `PersonalizedIndependent` / `NoPrice` / `Popularity` | Yes | Binomial-like tail only | No | None | None | Yes |

---

## 8. Optimization, Preconditioning, and Empirical Verification Summary

### 8.1 Diagonal Curvature Preconditioning & L-BFGS-B Stationarity
In retail panels with ribuan households ($|\mathcal{H}| = 2,495$ on Dunnhumby), the Hessian of the regularized negative log-likelihood

$$\mathcal{L}(\vartheta) = -\frac{1}{n_{\text{train}}} \sum_{i=1}^{n_{\text{train}}} \log P_\vartheta(y_i \mid z_i) + \frac{\lambda_{\text{ridge}}}{2 n_{\text{train}}} \|\vartheta_{\text{pen}}\|_2^2$$

is highly ill-conditioned because household parameters $H_h$ receive gradients proportional to household trip frequency $f_h = n_h / n_{\text{train}} \sim 4 \times 10^{-4}$, whereas dense global parameters ($g(N)$, $M^{\text{cat}}$, $\delta^{(1)}$) receive gradients from all $n_{\text{train}}$ baskets ($O(1)$ curvature).

We equalize parameter curvature via diagonal preconditioning $\vartheta = S \tilde{\vartheta}$ before invoking L-BFGS-B:
* **Household-indexed blocks** ($s_h, H^{\text{mix}}, H^{\text{cat}}, H^{(1)}$):
  $$S_{h, \cdot} = \left(0.15 f_h + \frac{\lambda_{\text{ridge}}}{n_{\text{train}}}\right)^{-1/2}.$$
* **Dense structural blocks** ($g(N), M^{\text{cat}}, M^{\text{prod-cat}}, \delta^{(1)}, L_1, R_1, \alpha^{\text{mix}}$):
  Scaled by $S_{\text{dense}} \in [10, 35]$ so that household and global directions advance at matched step sizes inside the L-BFGS-B two-loop recursion.

### 8.2 Summary of Empirical Results Across All Benchmarks
Full experimental tables and paired bootstrap confidence intervals are stored in [`results/benchmark_summary.json`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/benchmark_summary.json):

1. **Real ERIM ($28$ Products, [`erim_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/erim_comparison.csv))**:
   - `CompositeWholeLawEnergy` achieves **`test_ll = -3.36333`** and **`validation_ll = -3.40444`**, beating all 11 baselines with every paired household-cluster 95% CI strictly above zero:
     - vs. `structured_energy` (`-3.37973`): **`+0.01641`** (`[+0.0145, +0.0183]`)
     - vs. `pairwise_ar` (`-3.38518`): **`+0.02185`** (`[+0.0165, +0.0270]`)
     - vs. `shopper_factor_ar` (`-3.39166`): **`+0.02833`** (`[+0.0230, +0.0338]`)
     - vs. `logistic_normal_vae` (`-3.39392`): **`+0.03059`** (`[+0.0259, +0.0350]`)
     - vs. `independent_parent` (`-3.40289`): **`+0.03956`** (`[+0.0354, +0.0437]`)
     - vs. `neural_nade` (`-3.43290`): **`+0.06957`** (`[+0.0640, +0.0758]`)
2. **Real Dunnhumby ($64$ Products, [`dunnhumby_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/dunnhumby_comparison.csv))**:
   - `CompositeWholeLawEnergy` achieves **`test_ll = -6.83215`** and **`validation_ll = -6.41988`**, ranking **#1 across all 12 models** in Test LL, Validation LL, Recall@5 (**`0.47764`**), and Brier (**`0.02721`**), while cutting `size_TV` by **75%** relative to `structured_energy` (`0.08163` $\to$ **`0.02027`**):
     - vs. `neural_nade` (`-6.84261`): **`+0.01046`** (Validation gain **`+0.03371`**, Store-Week 95% CI `[+0.0014, +0.0189]`)
     - vs. `structured_energy` (`-6.94702`): **`+0.11487`** (`[+0.1062, +0.1240]`)
     - vs. `mission_esp` (`-7.07062`): **`+0.23847`** (`[+0.2246, +0.2522]`)
     - vs. `shopper_factor_ar` (`-7.09509`): **`+0.26294`** (`[+0.2452, +0.2806]`)
     - vs. `pairwise_ar` (`-7.10273`): **`+0.27058`** (`[+0.2526, +0.2873]`)
     - vs. `logistic_normal_vae` (`-7.14771`): **`+0.31556`** (`[+0.2961, +0.3327]`)
     - vs. `independent_parent` (`-7.30746`): **`+0.47531`** (`[+0.4503, +0.5020]`)
3. **Synthetic Counterfactual Worlds ([`synthetic_missions_a_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/synthetic_missions_a_comparison.csv) & [`synthetic_zero_price_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/synthetic_zero_price_comparison.csv))**:
   - `CalibratedMissionLaw` ranks **#1 across all 12 models** in Test LL (`-5.23719` on `missions_a`, `-5.17654` on `zero_price`), achieves `size_tv = 0.00197`, passes all 9 price-response gate checks with `own_price_rel_mae = 0.0807`, `same_category_rel_mae = 0.1994`, and **`cross_category_rel_mae = 0.1768`** ($\le 0.25$), and selects `6/6` optimal promotion plans.
