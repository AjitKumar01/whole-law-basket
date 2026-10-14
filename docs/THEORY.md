# Mathematical Theory and Architecture of the Whole-Law Basket Model

Implementation reference. For commands and current evidence status, see the [README](../README.md). The numerical identities below define the model; finite quadrature accuracy is supported by the stated tests rather than a uniform error guarantee. Historical benchmark tables are not results from the corrected fresh-fit runner.

This document provides a first-principles mathematical treatment of the **Whole-Law Basket Choice and Generation Architecture** implemented in [`whole_law_basket`](../whole_law_basket):
1. **Exact-Mode-Centered Analytic Hyperbolic-Sine ($\sinh$) Log-Time Quadrature for Exponential-Race Plackett–Luce (PL) Choice** ([`race_pl.py`](../whole_law_basket/race_pl.py)) for scalable within-category unordered subset choice across arbitrary category width $P$ and arbitrary heterogeneous utilities.
2. **`CompositeWholeLawEnergy`** ([`proposed_model.py`](../whole_law_basket/proposed_model.py)) for real retail scanner panels (ERIM and Dunnhumby), unifying category-frontier dynamic programming (DP), nonparametric global basket-size potentials, size-1 quick-trip stratification, polynomial Elementary Symmetric Polynomial (`log_esp`) within-category transitions, and price-invariant pre-trip category and product mission mixtures (`allow_price_mix=False`).
3. **`CalibratedMissionLaw`** ([`proposed_model.py`](../whole_law_basket/proposed_model.py)) for latent-mission counterfactual price response, combining differentiable race-PL subset choice with **joint permutation-equivariant hierarchical price regularization inside the L-BFGS-B training objective** (`regularization_loss` + `fit_with_validation_selection`) across heterogeneous synthetic worlds (`missions_a`, `missions_b`, and `zero_price`).
4. **Benchmark protocol and implementation**: see Section 5 for current fitting, checkpoint, and evaluation behavior.

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

---

## 2. Solution Part I: Exact-Mode Analytic Hyperbolic-Sine ($\sinh$) Quadrature for Exponential-Race Plackett–Luce Choice

### 2.1 Exponential-Race Representation of Unordered Plackett–Luce Choice
Let $u_c = (u_{c,1}, \dots, u_{c,P}) \in \mathbb{R}^P$ be the product utilities in category $c$, and define the normalized positive rates

$$w_i = \frac{\exp(u_{c,i})}{\sum_{m=1}^P \exp(u_{c,m})}, \qquad \sum_{i=1}^P w_i = 1.$$

By the Thurstonian / Gumbel-race construction of the Plackett–Luce model, drawing an ordered sequence without replacement according to rates $(w_1, \dots, w_P)$ is distributionally identical to assigning each product $i \in \{1, \dots, P\}$ an independent exponential arrival time $T_i \stackrel{\text{ind}}{\sim} \operatorname{Exponential}(w_i)$ and sorting products in increasing order of arrival times. A nonempty proper subset $A \subset \{1, \dots, P\}$ ($0 < k = |A| < P$) constitutes the exact set of the first $k$ chosen items if and only if every item in $A$ arrives strictly before every item in the complement $A^c = \{1, \dots, P\} \setminus A$:

$$\max_{i \in A} T_i < T_{\text{out}} := \min_{j \in A^c} T_j \sim \operatorname{Exponential}(w_{\text{out}}), \qquad w_{\text{out}} = \sum_{j \in A^c} w_j.$$

Conditioning on $T_{\text{out}} = \tau$ and applying the unit-rate change of variables $s = w_{\text{out}} \tau \in (0, \infty)$ with relative rates $r_i = w_i / w_{\text{out}} > 0$ yields the exact 1D integral identity:

$$P_{\text{PL}}(A \mid u_c) = \int_0^\infty e^{-s} \prod_{i \in A} \left(1 - e^{-r_i s}\right) ds.$$

### 2.2 Why Arithmetic-Mean Mode Scaling Fails Under Heterogeneous Utilities
When all selected items have equal utilities ($r_i \equiv \bar{r}$), the integrand $e^{-s}(1 - e^{-\bar{r} s})^k$ peaks near $s_0 = \frac{1}{\bar{r}}\log(1 + k\bar{r})$. However, when utilities within $A$ are strongly heterogeneous—for example, $P = 8$ products with $k = 7$ selected and utilities $u = (5, 5, 5, 5, 5, 5, -5, 0)$—the subset $A$ containing six items with $u_i = 5$ ($r_i = e^5 \approx 148.4$) and one low-utility item with $u_7 = -5$ ($r_7 = e^{-5} \approx 0.00674$) has arithmetic mean $\bar{r} \approx 127.2$, predicting a mode near $s_0 \approx 0.053$. In reality, the factor $(1 - e^{-r_7 s})$ vanishes near $s = 0.053$ and pushes the true integrand peak to $s^* \approx 0.997$—nearly **20 times larger** than $s_0$. Coupled with a fractional-power rational chart $s(t) = s_0 (t/(1-t))^{2/\sqrt{k}}$, non-integer exponents at $k = 7$ introduce branch-point singularities at $t \in \{0, 1\}$ that left $18.95\%$ of total subset probability mass missing ($\sum_{|A|=7} P(A) = 0.81049$ instead of $1.0$).

### 2.3 Exact Log-Time Mode Localization and Global Strict Concavity
To improve numerical accuracy for heterogeneous utilities and subset sizes, we change variables to log-time:

$$v = \log s \in (-\infty, \infty), \qquad s = e^v, \qquad ds = e^v dv.$$

In log-time $v$, the exact log-integrand is

$$\Phi(v) = v - e^v + \sum_{i \in A} \log\!\big(1 - e^{-r_i e^v}\big).$$

Define the smooth positive auxiliary function

$$\psi(x) = \frac{x}{e^x - 1} = \frac{x}{\operatorname{expm1}(x)} \in (0, 1) \quad (x > 0), \qquad \lim_{x \to 0^+} \psi(x) = 1.$$

Differentiating $\Phi(v)$ with respect to $v$ using $\frac{d}{dv}\log(1 - e^{-r_i e^v}) = \frac{r_i e^v e^{-r_i e^v}}{1 - e^{-r_i e^v}} = \psi(r_i e^v)$ gives the exact first and second derivatives:

$$\Phi'(v) = 1 - e^v + \sum_{i \in A} \psi(r_i e^v),$$

$$\Phi''(v) = -e^v + \sum_{i \in A} \psi(r_i e^v)\big(1 - r_i e^v - \psi(r_i e^v)\big).$$

**Theorem 1 (Global Strict Concavity and Analyticity of $\Phi(v)$)**:
For every $x > 0$, $\psi(x)(1 - x - \psi(x)) = \frac{x(e^x - 1 - x e^x)}{(e^x - 1)^2} < 0$ because $e^x(1 - x) < 1$ for all $x > 0$, and $\psi(x)(1-x-\psi(x)) = -x/2 + O(x^2)$ as $x \to 0^+$. Furthermore, $-e^v < 0$ for all $v \in \mathbb{R}$. Therefore:
1. $\Phi''(v) < -e^v < 0$ for all $v \in (-\infty, \infty)$, so $\Phi(v)$ is **globally strictly concave** on $\mathbb{R}$.
2. $\lim_{v \to -\infty} \Phi'(v) = 1 + k > 0$ and $\lim_{v \to +\infty} \Phi'(v) = -\infty$, so $\Phi(v)$ possesses a **unique global maximum** $v^* \in (-\infty, \infty)$ satisfying $\Phi'(v^*) = 0$.
3. Because $1 - e^{-r_i e^v} > 0$ for all real $v \in \mathbb{R}$, $\Phi(v)$ is **$C^\infty$ analytic** along the entire real axis with zero branch points.

In [`race_pl.py`](../whole_law_basket/race_pl.py), 14 damped Newton steps inside `torch.no_grad()` approximate the unique mode $v^*$ and its local scale $\sigma_v = (-\Phi''(v^*))^{-1/2}$.

### 2.4 Analytic Hyperbolic-Sine ($\sinh$) Gauss–Legendre Quadrature
We map the standard $M$-point Gauss–Legendre nodes $(\xi_q, \varpi_q)_{q=1}^M$ on $(-1, 1)$ (with $M = 128$) to a finite interval of log-time via the analytic hyperbolic-sine chart centered at $v^*$ and scaled by $\sigma_v$:

$$v_q = v^* + \alpha \sigma_v \sinh(\beta \xi_q), \qquad \frac{dv}{d\xi}(\xi_q) = \alpha \beta \sigma_v \cosh(\beta \xi_q), \qquad (\alpha = 2.6, \; \beta = 3.4).$$

Because $v^* \in \mathbb{R}$ and $\sigma_v > 0$ are detached (`torch.no_grad()`), they define a fixed coordinate chart for each basket/category slice, so reverse-mode automatic differentiation flows cleanly through the exact integrand evaluation:

$$\boxed{
\log P_{\text{PL}}(y_c \mid u_c) = \operatorname{logsumexp}_{q=1}^M \left[ \log \varpi_q + \log(\alpha \beta \sigma_v \cosh(\beta \xi_q)) + v_q - e^{v_q} + \sum_{p=1}^P y_{c,p} \log\!\big(-\operatorname{expm1}(-r_p e^{v_q})\big) \right].
}$$

### 2.5 Empirical Verification Across Large $(P, k)$ and Extreme Heterogeneous Utilities
The complete verification suite (`verify_race_pl_quadrature()` in [`race_pl.py`](../whole_law_basket/race_pl.py), saved in [`results/quadrature_verification.json`](../results/historical/quadrature_verification.json)) confirms machine precision across all regimes:
* **Exhaustive $P = 4$ subset & gradient check**: Maximum log-probability error across all $16$ subsets vs. exact $2^P$ DP is **`8.05e-13`**; maximum autograd gradient error is **`1.76e-13`**.
* **Large-category equal-utility combinatorial checks**:
  - $(P=16, k=3)$: max error **`4.44e-15`**
  - $(P=64, k=3)$: max error **`5.33e-15`**
  - $(P=64, k=32)$: max error **`7.11e-15`**
  - $(P=64, k=63)$: max error **`4.44e-15`**
  - $(P=1024, k=3)$: max error **`3.55e-15`**
  - $(P=1024, k=512)$: max error **`0.0`** (shift-invariance gradient sum error **`5.68e-14`**)
* **Heterogeneous-utility normalization stress tests ($P = 8, k = 7$, utilities `[s, s, s, s, s, s, -s, 0]`)**:

| Utility Scale $s$ | Previous Quadrature Sum | Exact-Mode $\sinh$ Quadrature Sum | Absolute Normalization Error $\big|\sum_{|A|=7} P(A) - 1\big|$ |
| :---: | :---: | :---: | :---: |
| $s = 2.0$ | `0.99220278` | `1.0000000000003317` | **`3.32e-13`** |
| $s = 5.0$ | `0.81049003` | `1.0000000000414049` | **`4.14e-11`** |
| $s = 10.0$ | `0.80385094` | `0.9999999999767986` | **`2.32e-11`** |
| $s = 20.0$ | `0.80384615` | `1.0000000000004399` | **`4.40e-13`** |

---

## 3. Solution Part II: `CompositeWholeLawEnergy` for Real Scanner Panels

To resolve the real-data transfer bottleneck on ERIM ($C=7, P=4, J=28$) and Dunnhumby ($C=16, P=4, J=64$), `CompositeWholeLawEnergy` ([`proposed_model.py`](../whole_law_basket/proposed_model.py)) constructs an **exact-normalized whole-basket mixture** over $\mathcal{S}_+ = \{0, 1\}^J \setminus \{\mathbf{0}\}$ with three structural guarantees:
1. **Price-Invariant Pre-Trip Mission Prior (`allow_price_mix=False`)**:
   Pre-trip shopping mission weights depend solely on household identity $h$ and global mission intercepts, never on in-store shelf prices $d_z$:
   $$\log \pi(z) = \operatorname{log\_softmax}\!\Big(\alpha^{\text{mix}} + H_h^{\text{mix}}\Big).$$
   Shelf prices enter exclusively through conditional item utilities $u(z)$, size-1 quick-trip utilities $u^{(1)}(z)$, and conditional category participation shifts $\Delta_c(z) = \kappa_c^{\text{shift}} + H_{h,c}^{\text{cat}} + (d_z \Phi^{\text{cat}})_c$.
2. **Polynomial `log_esp` Within-Category Transitions ($O(K P^2)$ vs. $O(K 2^P)$)**:
   For any category $c$ without intra-category edge couplings ($\mathcal{E}_{\text{within}, c} = \emptyset$), the inclusive nonempty log-weight over all $2^P - 1$ subsets is evaluated in $O(P^2)$ time via Elementary Symmetric Polynomials (`log_esp`) rather than $O(2^P)$ subset enumeration:
   $$\log W_c(u_c) = \operatorname{logsumexp}_{n=1}^P \left[ \log E_n\!\big(\exp(u_c)\big) + \gamma_c \frac{n(n-1)}{2} \right].$$
3. **Three Exact-Normalized Sub-Laws ($P_0$, $\{P_k^{\text{cat}}\}_{k=1}^{K_c}$, $\{P_m^{\text{prod}}\}_{m=1}^{K_p}$)**:
   - **Component $0$ ($P_0$)**: Combines the inherited category-frontier DP partition $W_{\text{base}}(N \mid z)$ for $N \ge 2$ with a nonparametric count potential $g(N)$, household count slope $s_h(N-1)$, and a dedicated size-1 quick-trip stratum $W_1(z) = \operatorname{logsumexp}_{j=1}^J u_j^{(1)}(z)$.
   - **Category Mission Components ($P_k^{\text{cat}}$)**: Couple category incidence across all $C$ categories via prototype vectors $M_k^{\text{cat}} \in \mathbb{R}^C$ with exact nonempty partition $\log Z_k^{\text{cat}}(z) = A_k^{\text{cat}}(z) + \log(-\operatorname{expm1}(-A_k^{\text{cat}}(z)))$.
   - **Product Bundle Mission Components ($P_m^{\text{prod}}$)**: Shift item utilities by SVD product bundle prototypes $M_m^{\text{prod}} \in \mathbb{R}^J$ and category modulations $M_{m,c}^{\text{prod-cat}}$ with exact nonempty partition $\log Z_m^{\text{prod}}(z)$.

---

## 4. Solution Part III: Nonempty-Conditioned Fisher Information and Joint Permutation-Equivariant Hierarchical Price Regularization in `CalibratedMissionLaw`

### 4.1 Exact $2 \times 2$ Nonempty-Conditioned Fisher Information Matrix
Consider $C = 2$ independent Bernoulli categories with natural parameters $(\eta_1, \eta_2)$, unconditional probabilities $p_c = \sigma(\eta_c) \in (0, 1)$, and $q_c = 1 - p_c$. Conditioning on a nonempty basket ($x \in \{0, 1\}^2 \setminus \{(0, 0)\}$) yields a natural exponential family with sufficient statistics $(x_1, x_2)$ and log-partition $\log Z_+(\eta_1, \eta_2) = \log(e^{\eta_1} + e^{\eta_2} + e^{\eta_1 + \eta_2})$:

$$\ell(\eta_1, \eta_2) = x_1 \eta_1 + x_2 \eta_2 - \log\!\big(e^{\eta_1} + e^{\eta_2} + e^{\eta_1 + \eta_2}\big).$$

Because this is a natural exponential family on $\{0, 1\}^2 \setminus \{(0, 0)\}$, the Fisher information matrix with respect to $(\eta_1, \eta_2)$ is the conditional covariance matrix $\operatorname{Cov}(x \mid x \neq \mathbf{0})$:

$$\mathcal{I}_\eta = -\nabla_{\eta \eta}^2 \ell(\eta) = \begin{pmatrix} \frac{p_1 q_1 (1 - q_1 q_2 - p_1 q_2)}{(1 - q_1 q_2)^2} & -\frac{p_1 q_1 p_2 q_2}{(1 - q_1 q_2)^2} \\ -\frac{p_1 q_1 p_2 q_2}{(1 - q_1 q_2)^2} & \frac{p_2 q_2 (1 - q_1 q_2 - q_1 p_2)}{(1 - q_1 q_2)^2} \end{pmatrix}.$$

At $p_1 = p_2 = 1/2$ ($\eta_1 = \eta_2 = 0$), the three nonempty outcomes $(1, 0), (0, 1), (1, 1)$ each have conditional probability $1/3$, so $\mathbb{E}[x_c \mid x \neq \mathbf{0}] = 2/3$, $\operatorname{Var}(x_c \mid x \neq \mathbf{0}) = 2/9$, and $\operatorname{Cov}(x_1, x_2 \mid x \neq \mathbf{0}) = 1/3 - 4/9 = -1/9$:

$$\mathcal{I}_\eta = \begin{pmatrix} 2/9 & -1/9 \\ -1/9 & 2/9 \end{pmatrix}, \qquad \mathcal{I}_\eta^{-1} = \begin{pmatrix} 6 & 3 \\ 3 & 6 \end{pmatrix}.$$

Observe that:
1. The **off-diagonal entry of the Fisher information matrix $\mathcal{I}_{\eta, 12} = -\frac{p_1 q_1 p_2 q_2}{(1 - q_1 q_2)^2} < 0$ is strictly negative** (at $p_1 = p_2 = 1/2$, $\mathcal{I}_{\eta, 12} = -1/9$), and the **off-diagonal entry of its inverse $\mathcal{I}_\eta^{-1}$ is strictly positive** (at $p_1 = p_2 = 1/2$, $(\mathcal{I}_\eta^{-1})_{12} = +3$).
2. Conditioning on $x \neq \mathbf{0}$ inflates the marginal variance of $\eta_c$ from $4$ (unconditional Bernoulli($0.5$)) to $6$ ($1.5\times$ inflation), and reduces the Fisher information along the common-mode eigenvector $v_+ = (1, 1)/\sqrt{2}$ from $1/4$ to $2/9 - 1/9 = 1/9$ (a $2.25\times$ variance inflation).

### 4.2 Why Price Blocks Must Be Estimated Jointly Under Permutation-Equivariant Hierarchical Penalties
In `CalibratedMissionLaw`, five interacting price parameter blocks enter the log-likelihood:
* Product-level own-price sensitivity `price` ($\beta_{c,p}^{\text{item}} \in \mathbb{R}^{C \times P}$),
* Category-incidence linear and hinge price slopes `category_price` ($\beta_c^{\text{cat}} \in \mathbb{R}^C$) and `category_hinge` ($\beta_c^{\text{hinge}} \in \mathbb{R}^C$) acting on $\bar{d}_{n,c} = \frac{1}{P}\sum_{p=1}^P d_{n,c,p}$ and $\max(\bar{d}_{n,c}, 0)$,
* Conditional category-count price slope `count_price` ($\beta_c^{\text{count}} \in \mathbb{R}^C$), and
* Cross-category price slope `category_cross` ($\gamma_c^{\text{cross}} \in \mathbb{R}^C$).

Two structural requirements govern the estimation of these blocks:
1. **Strict Category-Permutation Invariance (No Category-Index Ordering)**:
   Category indices $c \in \{0, \dots, C-1\}$ in retail catalogues are arbitrary labels without intrinsic economic ordering. Any regularizer that shrinks parameters toward an ordered function of $c$ violates permutation equivariance under relabeling $\pi \in S_C$.
2. **Joint Estimation Inside the L-BFGS-B Objective (No Post-Hoc Block Modification)**:
   Because $\bar{d}_{n,c}$ and $\max(\bar{d}_{n,c}, 0) = \frac{1}{2}\bar{d}_{n,c} + \frac{1}{2}|\bar{d}_{n,c}|$ have correlation $0.8632$ and interact directly with item-level $\beta_{c,p}^{\text{item}}$ and count slope $\beta_c^{\text{count}}$, modifying one block post-hoc after the other blocks have already adapted to it distorts the joint likelihood equilibrium.

Instead, `CalibratedMissionLaw.regularization_loss` ([`proposed_model.py`](../whole_law_basket/proposed_model.py)) embeds a **strictly permutation-invariant hierarchical quadratic penalty** directly inside the L-BFGS-B objective (`fit_lbfgsb`):

$$\mathcal{R}_{\text{price}}(\theta) = \sum_{c=1}^C \sum_{p=1}^P m_{c,p}^{\text{item}} (\beta_{c,p}^{\text{item}})^2 + \sum_{b \in \{\text{cat}, \text{hinge}, \text{count}, \text{cross}\}} w_b \sum_{c=1}^C m_{b,c} (\beta_{b,c})^2 + \bar{m}^{\text{item}}\left[ w_{\text{within}} \sum_{c=1}^C \sum_{p=1}^P (\beta_{c,p}^{\text{item}} - \bar{\beta}_{c,\cdot}^{\text{item}})^2 + w_{\text{between}} P \sum_{c=1}^C (\bar{\beta}_{c,\cdot}^{\text{item}} - \bar{\beta}_{\cdot,\cdot}^{\text{item}})^2 \right] + \bar{m}^{\text{cat}} \sum_{b \in \{\text{cat}, \text{hinge}, \text{count}\}} w_{\text{pool}, b} \sum_{c=1}^C (\beta_{b,c} - \bar{\beta}_{b,\cdot})^2,$$

where $\bar{\beta}_{c,\cdot}^{\text{item}} = \frac{1}{P}\sum_{p=1}^P \beta_{c,p}^{\text{item}}$, $\bar{\beta}_{\cdot,\cdot}^{\text{item}} = \frac{1}{CP}\sum_{c,p} \beta_{c,p}^{\text{item}}$, and $\bar{\beta}_{b,\cdot} = \frac{1}{C}\sum_{c=1}^C \beta_{b,c}$. Every term in $\mathcal{R}_{\text{price}}(\theta)$ is a symmetric sum over categories $c \in \{1, \dots, C\}$ and items $p \in \{1, \dots, P\}$, guaranteeing **exact invariance under any category permutation $\pi \in S_C$ and within-category item permutation $\sigma_c \in S_P$** (verified to `< 1e-12` in `test_category_permutation_invariance_of_regularization()` in [`tests/test_whole_law.py`](../tests/test_whole_law.py)). `CalibratedMissionLaw.fit_with_validation_selection` optimizes candidate hyperparameter configurations (`unregularized`, `mild_hierarchical`, `moderate_hierarchical`, `strong_hierarchical`) jointly to certified L-BFGS-B stationarity ($\le 2 \times 10^{-4}$) and selects the model maximizing held-out validation log-likelihood.

---

## 5. Training, evaluation, and evidence limits

The current runner trains correction models from constructor/training-data initializations. Reference `*_init.pt` files are never loaded. Real-data preference controls and the screened graph remain explicit shared inputs; this is not complete raw-transaction-to-model retraining.

The real composite fits its structured-energy dependency first. Its nested base is then frozen by architectural design. The dependency cannot be overwritten by an old composite checkpoint. Active and frozen parameter names are recorded; stationarity refers to active parameters only.

The objective is mean training negative log-likelihood plus the configured regularization. L-BFGS-B has a declared iteration limit and dual raw/scaled projected-gradient thresholds of `2e-4`. Certificates are recomputed on the actual saved weights after count refinement. Only converged regularization candidates are eligible for validation selection. Numerical stationarity does not certify a global optimum or a causal interpretation.

Resume requires matching content fingerprints for prepared splits, contexts, configuration, fixed state, dependencies, source code, library versions, fitting policy, and checkpoint bytes. Evaluation always runs again from loaded weights. Real-data generation uses 48 draws on identical test contexts; synthetic evaluation exactly integrates the 12-product basket distribution. Promotion evaluation uses explicitly hypothetical costs and funding.

Historical tables are under [`results/historical`](../results/historical). They precede the corrected fresh-fit policy, include unconverged baselines and warm-started fits, and do not establish a clean final ranking. New outputs default to `results/verified_v2/`. The corrected full suite remains to be run.

The synthetic price law and the real-data composite are different architectures. The synthetic cross-category law uses a specified cyclic neighbor graph; penalty permutation invariance does not make that graph arbitrary. Pooling, the default count/participation coherence ratio, coefficient constraints, and price-invariant mission weights are modeling assumptions. Independent intervention evidence is needed to validate real cross-price responses and distinguish complementarity from shared shopping missions.

Both proposed models describe nonempty binary baskets within a selected catalogue. Trip arrival, quantities, stockouts, and purchases outside that catalogue are outside this prediction target. Within-category scaling tests are not evidence of an end-to-end full-catalogue real-data benchmark.
