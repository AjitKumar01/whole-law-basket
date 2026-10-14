# A Complete Textbook on the Whole-Law Basket Choice and Generation Model

This textbook explains the **Whole-Law Basket Model** and every mathematical, statistical, and computational idea behind the [`whole_law_basket`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket) codebase from first principles. Every concept is introduced in plain, everyday language first, illustrated with a concrete example, and then connected to its exact mathematical formula and Python implementation.

---

## Table of Contents

1. [Chapter 1: What Problem Are We Solving? (The Supermarket Basket Problem in Plain Words)](#chapter-1-what-problem-are-we-solving-the-supermarket-basket-problem-in-plain-words)
2. [Chapter 2: The Anatomy of a Shopping Trip (Notation, Controls, and Utilities)](#chapter-2-the-anatomy-of-a-shopping-trip-notation-controls-and-utilities)
3. [Chapter 3: How to Pick $k$ Items Out of $P$ Without Order (`log_esp` and Exponential-Race Plackett–Luce)](#chapter-3-how-to-pick-k-items-out-of-p-without-order-log_esp-and-exponential-race-plackettluce)
4. [Chapter 4: Architecture I — `CalibratedMissionLaw` for Price & Promotion Counterfactuals](#chapter-4-architecture-i--calibratedmissionlaw-for-price--promotion-counterfactuals)
5. [Chapter 5: Architecture II — `StructuredEnergy` and `CompositeWholeLawEnergy` for Real Scanner Panels](#chapter-5-architecture-ii--structuredenergy-and-compositewholelawenergy-for-real-scanner-panels)
6. [Chapter 6: All 11 Baseline Models Explained Simply](#chapter-6-all-11-baseline-models-explained-simply)
7. [Chapter 7: Optimization, Curvature Preconditioning, and Stationarity Certificates](#chapter-7-optimization-curvature-preconditioning-and-stationarity-certificates)
8. [Chapter 8: How We Test Basket Models (Likelihood, Generation, Counterfactuals, and Promotions)](#chapter-8-how-we-test-basket-models-likelihood-generation-counterfactuals-and-promotions)
9. [Chapter 9: Empirical Results Across ERIM, Dunnhumby, and Synthetic Worlds](#chapter-9-empirical-results-across-erim-dunnhumby-and-synthetic-worlds)
10. [Chapter 10: Codebase Map and Annotated References](#chapter-10-codebase-map-and-annotated-references)

---

## Chapter 1: What Problem Are We Solving? (The Supermarket Basket Problem in Plain Words)

### 1.1 Single-Item Choice vs. Whole-Basket Choice
In classical discrete choice theory—pioneered by [Luce (1959)](https://www.statisticshowto.com/plackett-luce-model/) and [McFadden (1974)](https://escholarship.org/uc/item/61s3q2xr)—a consumer walks into a store and picks **exactly one item** from a mutually exclusive set of alternatives (for example, choosing one commuter train route or one brand of milk). The standard Multinomial Logit (MNL) model assigns each item $j$ a utility $u_j$ and says the probability of picking item $j$ is $\exp(u_j) / \sum_{m} \exp(u_m)$.

Real supermarket shopping does not work that way. On a single trip to a grocery store:
* A shopper might buy **1 item** (grabbing coffee on the way to work) or **15 items** (a weekly family stock-up trip).
* Within a single aisle (say, the yogurt category), the shopper might buy **0 items**, **1 item**, or **3 different flavors at once**.
* Across different aisles, categories are linked by **shopping missions** ([Manchanda, Ansari & Gupta 1999](https://pubsonline.informs.org/doi/10.1287/mksc.18.2.95); [Russell & Petersen 2000](https://www.researchgate.net/publication/343484851_Market_Basket_Analysis_Recommendation_System_Using_Association_Rules)): a shopper buying pasta is much more likely to buy pasta sauce in the next aisle, and a shopper on a "quick breakfast run" has a very different basket-size distribution than the same shopper on a Sunday stock-up trip.

### 1.2 The Sample Space of Nonempty Binary Baskets ($\mathcal{S}_+$)
Suppose a retailer tracks $C$ product categories (aisles), each containing $P$ products, for a total catalogue of $J = C \times P$ products:
* In our **synthetic worlds** ([`whole_law_basket/synthetic.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/synthetic.py)), $C = 3$ categories $\times$ $P = 4$ products $= J = 12$ products.
* In the **ERIM scanner panel** ([`whole_law_basket/real_data.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/real_data.py)), $C = 7$ categories $\times$ $P = 4$ products $= J = 28$ products.
* In the **Dunnhumby Complete Journey panel** ([`whole_law_basket/real_data.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/real_data.py)), $C = 16$ categories $\times$ $P = 4$ products $= J = 64$ products.

A shopping basket is a binary table $y \in \{0, 1\}^{C \times P}$ (or flattened vector $y \in \{0, 1\}^J$), where $y_{c,p} = 1$ if product $p$ in category $c$ is in the basket and $y_{c,p} = 0$ if it is not.

Crucially, checkout scanners only record a transaction when **at least one item is purchased**. An all-zero vector $y = \mathbf{0}$ is never observed in checkout data. Therefore, the set of possible baskets is the set of all **nonempty binary vectors**:

$$\mathcal{S}_+ = \{0, 1\}^J \setminus \{\mathbf{0}\}, \qquad |\mathcal{S}_+| = 2^J - 1.$$

### 1.3 The "Curse of $2^J - 1$"
To define a valid probability distribution $P(y \mid x)$ over all nonempty baskets $y \in \mathcal{S}_+$, the probabilities of all $2^J - 1$ nonempty baskets must sum to exactly $1$:

$$\sum_{y \in \{0, 1\}^J \setminus \{\mathbf{0}\}} P(y \mid x) = 1.$$

Look at how fast $2^J - 1$ grows:
* For $J = 12$ (synthetic): $2^{12} - 1 = 4,095$ baskets (small enough to list exhaustively on a computer).
* For $J = 28$ (ERIM): $2^{28} - 1 = 268,435,455$ baskets ($\approx 268\text{ million}$ baskets—too large to sum over for every training trip).
* For $J = 64$ (Dunnhumby): $2^{64} - 1 \approx 1.84 \times 10^{19}$ baskets ($18.4\text{ quintillion}$ baskets—impossible to enumerate).

### 1.4 How Previous Models Tried to Dodge the $2^J - 1$ Sum—and What Went Wrong
Historically, researchers used three shortcuts to avoid summing over $2^J - 1$ baskets, each introducing a major flaw:

1. **Shortcut 1: Assume Every Product Is an Independent Coin Flip (`PersonalizedIndependent`)**
   If each product $j$ is bought independently with probability $p_j = \sigma(u_j) = (1 + e^{-u_j})^{-1}$, then the probability of the all-zero basket is $\prod_{j=1}^J (1 - p_j)$, so conditioning on $y \neq \mathbf{0}$ is easy.
   * *Why it fails*: In real life, products within a category are **substitutes** (if you buy one brand of ketchup, you rarely buy three other brands of ketchup on the same trip), whereas categories across the store are driven by **trip missions** and **budget/carrying constraints**. Independent coin flips severely mispredict both within-category cannibalization and total basket size $N = \sum_j y_j$.

2. **Shortcut 2: Pairwise Markov Random Fields / Pseudolikelihood ([Besag 1974](https://academic.oup.com/jrsssb/article/36/2/192/7027394); [Russell & Petersen 2000](https://www.researchgate.net/publication/343484851_Market_Basket_Analysis_Recommendation_System_Using_Association_Rules))**
   Write an Ising-like energy $\mathcal{E}(y) = \sum_j u_j y_j + \sum_{i < j} W_{ij} y_i y_j$ and estimate $W_{ij}$ by predicting one item given all other items ($P(y_j \mid y_{-j})$, called *pseudolikelihood*).
   * *Why it fails*: Pseudolikelihood never computes the normalizing constant $Z = \sum_{y \neq \mathbf{0}} \exp(\mathcal{E}(y))$. Without $Z$, you cannot evaluate held-out basket log-likelihoods, you cannot compute exact counterfactual basket probabilities when a price changes, and generating baskets requires slow Markov chain Monte Carlo (MCMC) sampling.

3. **Shortcut 3: Sequential / Autoregressive Chain Rule ([Uria et al. 2016 (NADE)](https://jmlr.org/papers/v17/16-272.html); `PairwiseBinaryAR`; `LowRankBilinearAR`)**
   Number the products $j = 1, 2, \dots, J$ in some fixed list order and factorize the basket probability step-by-step:
   $$P(y \mid x) \propto P(y_1 \mid x) \cdot P(y_2 \mid y_1, x) \cdot P(y_3 \mid y_1, y_2, x) \cdots P(y_J \mid y_1, \dots, y_{J-1}, x).$$
   * *Why it fails*: A shopping basket at checkout is an **unordered set**, not a sentence with a beginning and an end. Forcing a fixed product order $1, 2, \dots, J$ means that whether product $5$ is bought changes the probability of product $6$, whereas product $6$ can only affect product $5$ indirectly through the normalizing constant. Worse, in neural autoregressive models (`FastNeuralNADE`), feeding the entire price vector $d$ into the hidden state allows shelf prices to shift global trip propensity in opaque, non-causal ways, leading to poor counterfactual promotion decisions (only `3/6` or `4/6` optimal promotion plans on our synthetic worlds).

### 1.5 What Is the Whole-Law Basket Model?
The **Whole-Law Basket Model** ([`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) is a **simultaneous, order-invariant, exactly normalized probability distribution over all $2^J - 1$ nonempty baskets**.
* **"Simultaneous & Order-Invariant"**: It never steps through products in an arbitrary sequence $j = 1 \dots J$. Renumbering products within a category or renumbering categories leaves the model family completely unchanged.
* **"Whole-Law"**: Instead of modeling only single-category choice or only pairwise conditionals, it models the **entire joint law of the basket**—pre-trip mission, which categories are visited ($x_c$), how many items are bought in each category ($n_c$) and in the whole basket ($N$), and which unordered subset of items ($y_c$) is chosen within each category.
* **"Exactly Normalized & Rejection-Free"**: Through three mathematical tools—**Elementary Symmetric Polynomials (`log_esp`)**, **Category-Frontier Dynamic Programming**, and **Exact-Mode $\sinh$ Log-Time Quadrature for Exponential-Race Plackett–Luce**—it computes the exact partition function over all $2^J - 1$ baskets and draws exact synthetic baskets in milliseconds without MCMC or rejection sampling.

---

## Chapter 2: The Anatomy of a Shopping Trip (Notation, Controls, and Utilities)

Before building the model, let's establish the exact building blocks of a single shopping trip $n \in \{1, \dots, B\}$.

### 2.1 Structural Summaries of a Basket $y$
Given a binary basket matrix $y \in \{0, 1\}^{C \times P}$:
1. **Category Count ($n_c$)**: How many distinct products the shopper bought in category $c$:
   $$n_c(y) = \sum_{p=1}^P y_{c,p} \in \{0, 1, \dots, P\}.$$
2. **Category Incidence ($x_c$)**: Did the shopper buy *anything at all* in category $c$?
   $$x_c(y) = \mathbb{I}\big(n_c(y) > 0\big) \in \{0, 1\}.$$
3. **Total Basket Size ($N$)**: How many distinct products the shopper bought across the entire store:
   $$N(y) = \sum_{c=1}^C n_c(y) = \sum_{c=1}^C \sum_{p=1}^P y_{c,p} \in \{1, \dots, J\}.$$

### 2.2 Trip Context and Observable Controls ($z$)
For every trip, the context dictionary `ctx` provides:
* `house` ($h \in \{0, \dots, U-1\}$): The household ID of the shopper.
* `store` ($s$): The store ID where the trip occurred.
* `week` ($t$): The calendar week of the trip (capturing seasonality via Fourier sine/cosine waves).
* `dlp` ($d_{c,p} \in \mathbb{R}$): The **de-meaned log price** of product $(c, p)$ on that trip relative to its training reference price:
  $$d_{c,p} = \log\!\left(\frac{p_{c,p}^{\text{offered}}}{p_{c,p}^{\text{ref}}}\right).$$
  When $d_{c,p} < 0$, the product is on **discount** (cheaper than usual). When $d_{c,p} > 0$, the product is **above its reference price**. When $d_{c,p} = 0$, it is at its normal reference price.
* `disp` ($D_{c,p} \in [0, 1]$) and `mail` ($M_{c,p} \in [0, 1]$): In-store promotional display and feature mailer indicators.

### 2.3 Price-Free Baseline Utility $b_{c,p}(z)$ and Net Product Utility $u_{c,p}(z)$
In [`whole_law_basket/real_data.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/real_data.py) (`TrainableBasketModel.baseline`), the price-free observable controls are combined into a baseline utility $b_{c,p}(z)$:

$$b_{c,p}(z) = \lambda_{c,p}^{\text{base}} + \theta_h^\top \alpha_{c,p} + \xi_s^\top \zeta_{c,p} + f(t)^\top \mu_{c,p} + w_{c,p}^D D_{c,p} + w_{c,p}^M M_{c,p},$$

where $\theta_h^\top \alpha_{c,p}$ captures low-rank household-product affinity, $\xi_s^\top \zeta_{c,p}$ captures store assortment effects, $f(t)^\top \mu_{c,p}$ captures seasonal Fourier harmonics, and $w^D, w^M$ capture display and mailer lifts.

Adding the model's item intercept $\tilde{\lambda}_{c,p}$, household preference $H_{h,c,p}$, and price response yields the **net product utility** $u_{c,p}(z)$:
* **Linear Own-Price Utility** (used in `CalibratedMissionLaw` and standard baselines):
  $$u_{c,p}(z) = b_{c,p}(z) + \tilde{\lambda}_{c,p} + H_{h,c,p} - \beta_{c,p} d_{c,p}.$$
* **Asymmetric Reference-Price Utility** (used in `StructuredEnergy` and `CompositeWholeLawEnergy` on real scanner panels):
  $$u_{c,p}(z) = b_{c,p}(z) + \tilde{\lambda}_{c,p} + H_{h,c,p} - \beta_{c,p}^- \min(d_{c,p}, 0) - \beta_{c,p}^+ \max(d_{c,p}, 0).$$
  Why split $d_{c,p}$ into $\min(d_{c,p}, 0)$ (discounts) and $\max(d_{c,p}, 0)$ (price increases)? Because in retail scanner data, shoppers often react differently to promotional price cuts ($\beta_{c,p}^-$) than to regular price increases ($\beta_{c,p}^+$). Note that when $d_{c,p} < 0$, $-\beta_{c,p}^- \min(d_{c,p}, 0) = +\beta_{c,p}^- |d_{c,p}| > 0$, which **increases** utility, whereas when $d_{c,p} > 0$, $-\beta_{c,p}^+ \max(d_{c,p}, 0) = -\beta_{c,p}^+ d_{c,p} < 0$, which **decreases** utility.

---

## Chapter 3: How to Pick $k$ Items Out of $P$ Without Order (`log_esp` and Exponential-Race Plackett–Luce)

Suppose a shopper has decided to buy $k$ distinct items in category $c$ (where $1 \le k \le P$). Which $k$ items do they pick, and how do we compute the probability of that unordered subset $A \subset \{1, \dots, P\}$ ($|A| = k$) without listing all $\binom{P}{k}$ combinations?

Our codebase uses two complementary, order-invariant mathematical mechanisms:
1. **Elementary Symmetric Polynomials (`log_esp`)** for Tilt/Energy models (`StructuredEnergy`, `CompositeWholeLawEnergy`, and ESP baselines).
2. **Exponential-Race Plackett–Luce (`race_pl.py`)** for Thurstonian random-utility mission models (`CalibratedMissionLaw`).

Let's understand both in plain words.

### 3.1 Mechanism 1: Elementary Symmetric Polynomials (`log_esp`)
Suppose category $c$ has $P = 4$ products with positive weights $v_p = \exp(u_{c,p})$ for $p \in \{1, 2, 3, 4\}$.
If the probability of picking a subset $A \subset \{1, \dots, P\}$ of size $k$ is proportional to the product of the weights of the items in $A$,

$$P(y_c = A \mid |y_c| = k) = \frac{\prod_{p \in A} \exp(u_{c,p})}{E_k(e^{u_c})} = \frac{\exp\!\left(\sum_{p \in A} u_{c,p}\right)}{E_k(e^{u_c})},$$

then the normalizing constant in the denominator is the **$k$-th Elementary Symmetric Polynomial (ESP)**:

$$E_k(v_1, \dots, v_P) = \sum_{\substack{A \subset \{1, \dots, P\} \\ |A| = k}} \prod_{p \in A} v_p.$$

#### Worked Example ($P = 3$)
For $P = 3$ items with weights $(v_1, v_2, v_3)$:
* $E_0 = 1$ (the empty set has product $1$).
* $E_1 = v_1 + v_2 + v_3$ (sum of singletons).
* $E_2 = v_1 v_2 + v_1 v_3 + v_2 v_3$ (sum of pairs).
* $E_3 = v_1 v_2 v_3$ (the full set of all 3 items).

#### How `log_esp` Computes All $E_0, E_1, \dots, E_P$ in $O(P^2)$ Time
Listing all $\binom{P}{k}$ subsets takes $O(2^P)$ time, which explodes when $P$ is large. However, notice that $(1 + v_1 x)(1 + v_2 x)\cdots(1 + v_P x) = \sum_{k=0}^P E_k x^k$. When we reveal one more item $j$ with weight $v_j = e^{u_j}$, a subset of size $k$ using the first $j$ items either:
* **does not include item $j$** (so all $k$ items came from the first $j-1$ items: $E_k^{(j-1)}$), or
* **does include item $j$** (so the other $k-1$ items came from the first $j-1$ items: $v_j \cdot E_{k-1}^{(j-1)}$).

In log-space ($\ell_k^{(j)} = \log E_k^{(j)}$), this recurrence is a single `torch.logaddexp` call in `log_esp` ([`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)):

$$\ell_k^{(j)} = \operatorname{logaddexp}\!\Big(\ell_k^{(j-1)}, \; \ell_{k-1}^{(j-1)} + u_j\Big), \qquad k = 1, \dots, j.$$

This computes all $\log E_0, \log E_1, \dots, \log E_P$ in **$O(P^2)$ additions** instead of $2^P$ subset evaluations!

#### How `_sample_esp_subset_local` Draws an Exact Subset of Size $k$ Without Rejection
Once we know the suffix ESP table $\ell_r^{(\text{suffix } j)}$ (the log-sum over subsets of size $r$ from items $\{j, \dots, P\}$), we can sample a subset of exact size $k$ in a single pass from $j = 1$ to $P$:
* If we still need $r$ items from $\{j, \dots, P\}$, the exact marginal probability that item $j$ is included in the size-$r$ subset is:
  $$P(y_j = 1 \mid \text{need } r \text{ items from } j \dots P) = \exp\!\Big(u_j + \ell_{r-1}^{(\text{suffix } j+1)} - \ell_r^{(\text{suffix } j)}\Big).$$
* If we flip a coin with this exact probability and take item $j$, we decrement $r \leftarrow r - 1$; otherwise we keep $r$ unchanged and move to $j+1$.
* Boundary conditions are automatic: if $r = 0$, the probability is $0$; if $r = P - j + 1$ (we need all remaining items), the probability is $1$. Every draw produces a subset of size **exactly $k$** on the first try!

---

### 3.2 Mechanism 2: Exponential-Race Plackett–Luce Choice ([`whole_law_basket/race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py))
While ESP models weight a subset by $\prod_{p \in A} e^{u_p}$, random-utility theory ([Luce 1959](https://www.statisticshowto.com/plackett-luce-model/); [McFadden 1974](https://escholarship.org/uc/item/61s3q2xr); [Plackett 1975](https://academic.oup.com/jrsssc/article/24/2/193/6953554)) models a shopper who has Gumbel-distributed random utilities $U_{c,p} = u_{c,p} + \varepsilon_{c,p}$ ($\varepsilon_{c,p} \stackrel{\text{i.i.d.}}{\sim} \operatorname{Gumbel}(0, 1)$) and picks their **top $k$ most-preferred items** in category $c$:

$$A = \operatorname{arg\,top}_k \{U_{c,1}, \dots, U_{c,P}\}.$$

Notice that the shopper does **not** care about the internal order among the $k$ items they put in their cart—only which unordered set $A$ of $k$ items beat the $P - k$ items left on the shelf! However, naively summing the Plackett–Luce permutation formula over all $k!$ orderings of $A$ requires $k!$ terms, which is impossible for large $k$.

#### The Continuous-Time Stopwatch Race Analogy
There is a classic equivalence between Gumbel maxima and exponential races:
1. Define the normalized positive rates $w_p = \frac{\exp(u_{c,p})}{\sum_{m=1}^P \exp(u_{c,m})}$ so $\sum_{p=1}^P w_p = 1$.
2. Give each product $p \in \{1, \dots, P\}$ an independent exponential stopwatch $T_p \stackrel{\text{ind}}{\sim} \operatorname{Exponential}(w_p)$, where higher utility $w_p$ means the product arrives **faster** (smaller $T_p$).
3. Sorting products from fastest arrival ($T_{(1)}$) to slowest arrival ($T_{(P)}$) generates an exact Plackett–Luce ranking!

Therefore, an unordered subset $A \subset \{1, \dots, P\}$ of size $k$ is the set of the top $k$ items **if and only if every item in $A$ arrives before the fastest item outside $A$** ($A^c = \{1, \dots, P\} \setminus A$):

$$\max_{i \in A} T_i < T_{\text{out}} := \min_{j \in A^c} T_j.$$

Since the minimum of independent exponentials is also exponential, the fastest outsider arrives at time

$$T_{\text{out}} \sim \operatorname{Exponential}(w_{\text{out}}), \qquad w_{\text{out}} = \sum_{j \in A^c} w_j.$$

If we condition on the fastest outsider arriving at time $\tau > 0$, each item $i \in A$ independently beats $\tau$ with probability $P(T_i < \tau) = 1 - e^{-w_i \tau}$. Averaging over $\tau \sim \operatorname{Exponential}(w_{\text{out}})$ and substituting $s = w_{\text{out}} \tau$ with relative rates $r_i = w_i / w_{\text{out}}$ collapses the $k!$ permutation sum into a **single 1D integral**:

$$\boxed{P_{\text{PL}}(A \mid u_c) = \int_0^\infty e^{-s} \prod_{i \in A} \left(1 - e^{-r_i s}\right) ds.}$$

### 3.3 Why Naive Quadrature Fails & How Exact-Mode $\sinh$ Log-Time Quadrature Solves It
How do we evaluate $\int_0^\infty e^{-s} \prod_{i \in A}(1 - e^{-r_i s})\,ds$ on a computer inside a neural/econometric optimizer so that both probabilities and reverse-mode `autograd` derivatives are accurate to 13–15 decimal digits?

#### The Heterogeneous-Utility Trap (Concrete Counterexample)
Suppose $P = 8$ products and $k = 7$ items are bought, with utilities $u = (5, 5, 5, 5, 5, 5, -5, 0)$. Consider the subset $A = \{1, 2, 3, 4, 5, 6, 7\}$ (leaving only item $8$ with $u_8 = 0$ outside, so $w_{\text{out}} = w_8$).
* Six items in $A$ have huge relative rates $r_1 = \dots = r_6 = e^5 \approx 148.4$.
* One item in $A$ has a tiny relative rate $r_7 = e^{-5} \approx 0.00674$.
* If you guess the peak of the integral using the **average rate** $\bar{r} = \frac{1}{7}\sum_{i \in A} r_i \approx 127.2$, you place your quadrature grid around $s_0 = \frac{1}{\bar{r}}\log(1 + 7\bar{r}) \approx 0.053$.
* **What goes wrong?** At $s = 0.053$, the slow item's factor $(1 - e^{-0.00674 s})$ is virtually zero! The actual peak of the integrand is pushed out to $s^* \approx 0.997$ (nearly $20\times$ larger than $s_0$). A quadrature grid centered at $s_0 = 0.053$ misses the peak completely, losing **18.95% of the total probability mass** ($\sum_{|A|=7} P(A) = 0.81049$ instead of $1.0$).

#### Our Solution in [`whole_law_basket/race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py)
We change variables to **log-time** $v = \log s \in (-\infty, \infty)$ ($s = e^v, ds = e^v dv$). In log-time, the log-integrand is:

$$\Phi(v) = v - e^v + \sum_{i \in A} \log\!\big(1 - e^{-r_i e^v}\big).$$

Using $\psi(x) = \frac{x}{e^x - 1} \in (0, 1)$, its first two derivatives are:

$$\Phi'(v) = 1 - e^v + \sum_{i \in A} \psi(r_i e^v), \qquad \Phi''(v) = -e^v + \sum_{i \in A} \psi(r_i e^v)\big(1 - r_i e^v - \psi(r_i e^v)\big).$$

Because $\psi(x)(1 - x - \psi(x)) < 0$ for all $x > 0$ and $-e^v < 0$, **$\Phi''(v)$ is strictly negative everywhere on $(-\infty, \infty)$**! That means:
1. $\Phi(v)$ has a **single, unique bell-shaped peak $v^*$** for *any* utilities $u_c$ and *any* subset size $k$.
2. In `_find_log_s_mode` ([`race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py)), 14 damped Newton steps (`v = v - Phi'(v)/Phi''(v)`) find the exact peak $v^*$ to machine precision ($|\Phi'(v^*)| < 10^{-14}$) and compute the exact local width $\sigma_v = (-\Phi''(v^*))^{-1/2}$.
3. Inspired by exponential/hyperbolic-sine quadrature ([Takahasi & Mori 1974](https://ems.press/journals/prims/articles/2686)), we map $M = 128$ Gauss–Legendre nodes $\xi_q \in (-1, 1)$ directly around the true peak $v^*$:
   $$v_q = v^* + \alpha \sigma_v \sinh(\beta \xi_q), \qquad \frac{dv}{d\xi}(\xi_q) = \alpha \beta \sigma_v \cosh(\beta \xi_q), \qquad (\alpha = 2.6, \; \beta = 3.4).$$
4. Because $v^*$ and $\sigma_v$ are computed inside `torch.no_grad()`, they act as a fixed coordinate chart while PyTorch `autograd` differentiates cleanly through the integrand evaluation at $v_q$. On the extreme $P=8, k=7$ stress test above, all $\binom{8}{7}=8$ subset probabilities now sum to $1.0$ within **$4.14 \times 10^{-11}$**, and on $P = 1024, k = 512$ the error is **$0.0$** in IEEE-754 float64 ([`results/quadrature_verification.json`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/quadrature_verification.json)).

---

## Chapter 4: Architecture I — `CalibratedMissionLaw` for Price & Promotion Counterfactuals

When a retailer asks: *"If we discount Product 2 in Category 1 by 15%, how much will sales of Product 2 rise, how much will sales of other products in Category 1 fall (cannibalization), and how much will sales in Category 2 change?"*, we need a structural model that separates pre-trip missions from category visit decisions and within-category brand choice. That is `CalibratedMissionLaw` in [`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py).

### 4.1 The Three Simultaneous Layers of `CalibratedMissionLaw`
For each trip $n$ by household $h$:

1. **Layer 1: Pre-Trip Mission Mixture ($m \in \{0, 1\}$)**
   Each household $h$ has a probability $w_h = \sigma(\text{mission\_logit}_h)$ of being on Mission 0 vs. Mission 1 (for example, a "Focused/Quick Mission" vs. a "Broad Stock-Up Mission"). Notice that $w_h$ depends on the household $h$, **not** on today's shelf prices.

2. **Layer 2: Category Visit ($x_c \in \{0,1\}$) and Category Item Count ($n_c \in \{1, \dots, P\}$)**
   Let $\bar{d}_{n,c} = \frac{1}{P}\sum_{p=1}^P d_{n,c,p}$ be the average log-price of category $c$ on trip $n$.
   * **Category Incidence Logit ($\eta_{n,m,c}$)**: Under mission $m \in \{0, 1\}$, the propensity to visit category $c$ is:
     $$\eta_{n,m,c} = \text{need}_c + \text{house\_need}_{h,c} \pm \frac{1}{2}\text{mission\_contrast}_c - \beta_c^{\text{cat}} \bar{d}_{n,c} + \gamma_c^{\text{cross}} \bar{d}_{n,c-1} - \beta_c^{\text{hinge}} \max(\bar{d}_{n,c}, 0) + \tau_c \text{IV}_{n,c}.$$
     Here:
     - $\text{IV}_{n,c} = \log \sum_{p=1}^P \exp(u_{n,c,p})$ is the **inclusive value** (expected maximum utility) of the products inside category $c$, scaled by $\tau_c = \operatorname{softplus}(\text{inclusive\_scale}_c)$.
     - $\beta_c^{\text{cat}} \bar{d}_{n,c}$ and $\beta_c^{\text{hinge}} \max(\bar{d}_{n,c}, 0)$ capture linear and surcharge-hinge category price sensitivity.
     - $\gamma_c^{\text{cross}} \bar{d}_{n,c-1}$ captures cross-category price complementarity/substitution.
     Conditioned on at least one category being visited ($x \neq \mathbf{0}$), the category visit pattern $x \in \{0,1\}^C \setminus \{\mathbf{0}\}$ has exact log-probability:
     $$\log P(x \mid \eta_{n,m}) = \sum_{c=1}^C \Big(x_c \log\sigma(\eta_{n,m,c}) + (1 - x_c)\log\sigma(-\eta_{n,m,c})\Big) - \log\!\left(1 - \prod_{c=1}^C \sigma(-\eta_{n,m,c})\right).$$
   * **Conditional Category Count ($n_c \in \{1, \dots, P\}$ given $x_c = 1$)**: If category $c$ is visited ($x_c = 1$), how many distinct items $k \in \{1, \dots, P\}$ does the shopper buy in category $c$?
     Define $\count\_\eta_{n,m,c} = \count\_\text{need}_c - \beta_c^{\text{count}} \bar{d}_{n,c} \pm \frac{1}{2}\count\_\text{contrast}_c$. The log-weights for buying $k = 1, 2, \dots, P$ items in category $c$ are:
     $$\big(0, \;\; \count\_\eta_{n,m,c}, \;\; 2\,\count\_\eta_{n,m,c} + \delta_{c,1}, \;\; \dots, \;\; (P-1)\,\count\_\eta_{n,m,c} + \delta_{c,P-2}\big),$$
     normalized via `log_softmax` over $k \in \{1, \dots, P\}$.

3. **Layer 3: Within-Category Subset Selection via Exponential-Race PL**
   Given that $n_c = |y_c|$ items are bought in category $c$, the unordered subset $y_c \in \{0,1\}^P$ has exact log-probability `race_observed_log_probability(u_{n,c}, y_{n,c})` from Chapter 3.

Finally, a small independent fallback mixture weight $\pi_{\text{ind}} = 0.02 + 0.23\,\sigma(\text{mixture\_logit})$ mixes the mission law with an independent Bernoulli nonempty law to prevent zero-probability outliers.

### 4.2 Why Conditioning on Nonempty Baskets ($x \neq \mathbf{0}$) Couples Categories: The $2 \times 2$ Fisher Information Proof
Even before adding missions or cross-price terms, conditioning on a nonempty basket ($x \neq \mathbf{0}$) mathematically couples categories together!

Let's prove this with $C = 2$ categories having logits $(\eta_1, \eta_2)$ and unconditional probabilities $p_1 = \sigma(\eta_1), p_2 = \sigma(\eta_2)$ ($q_c = 1 - p_c$). On nonempty trips $\{ (1,0), (0,1), (1,1) \}$, the log-likelihood is:

$$\ell(\eta_1, \eta_2) = x_1 \eta_1 + x_2 \eta_2 - \log(e^{\eta_1} + e^{\eta_2} + e^{\eta_1 + \eta_2}).$$

Taking minus the second derivatives gives the **Fisher Information Matrix** $\mathcal{I}_\eta = \operatorname{Cov}(x \mid x \neq \mathbf{0})$:

$$\mathcal{I}_\eta = \begin{pmatrix} \frac{p_1 q_1 (1 - q_1)}{(1 - q_1 q_2)^2} & -\frac{p_1 q_1 p_2 q_2}{(1 - q_1 q_2)^2} \\ -\frac{p_1 q_1 p_2 q_2}{(1 - q_1 q_2)^2} & \frac{p_2 q_2 (1 - q_2)}{(1 - q_1 q_2)^2} \end{pmatrix}.$$

At $p_1 = p_2 = 1/2$ ($\eta_1 = \eta_2 = 0$), each of the 3 nonempty patterns $(1,0), (0,1), (1,1)$ has probability $1/3$:

$$\mathcal{I}_\eta = \begin{pmatrix} 2/9 & -1/9 \\ -1/9 & 2/9 \end{pmatrix}, \qquad \mathcal{I}_\eta^{-1} = \begin{pmatrix} 6 & 3 \\ 3 & 6 \end{pmatrix}.$$

**What does this mean in plain English?**
Because the $(0, 0)$ basket is missing from checkout data, observing $x_1 = 0$ *forces* $x_2 = 1$. This creates negative conditional covariance ($\mathcal{I}_{\eta, 12} = -1/9 < 0$) between $x_1$ and $x_2$, and positive covariance ($(\mathcal{I}_\eta^{-1})_{12} = +3 > 0$) between the maximum-likelihood estimates $(\hat{\eta}_1, \hat{\eta}_2)$, inflating estimation variance along the common direction $(1, 1)$ by $2.25\times$!

### 4.3 Why All Price Blocks Must Be Regularized Jointly with Permutation-Equivariant Hierarchical Penalties
In `CalibratedMissionLaw`, five price parameter blocks interact on every trip:
1. Product own-price `price` ($\beta_{c,p}^{\text{item}}$),
2. Category linear price `category_price` ($\beta_c^{\text{cat}}$),
3. Category surcharge hinge `category_hinge` ($\beta_c^{\text{hinge}}$),
4. Category count price `count_price` ($\beta_c^{\text{count}}$), and
5. Cross-category price `category_cross` ($\gamma_c^{\text{cross}}$).

Notice two critical facts about these blocks:
* **Collinearity between $\bar{d}_{n,c}$ and $\max(\bar{d}_{n,c}, 0)$**: Because $\max(\bar{d}, 0) = \frac{1}{2}\bar{d} + \frac{1}{2}|\bar{d}|$, the linear term $\bar{d}_{n,c}$ and the hinge term $\max(\bar{d}_{n,c}, 0)$ have a correlation of **$0.8632$**! Furthermore, the conditional variance of $\max(\bar{d}_{n,c}, 0)$ given $\bar{d}_{n,c}$ is $46\times$ smaller than product-level price variance. In unregularized estimation (as on `missions_b`), sampling noise can shift slope from $\beta_c^{\text{cat}}$ onto $\beta_c^{\text{hinge}}$, which distorts how the category responds to price *discounts* ($\bar{d}_{n,c} < 0$, where the hinge is zero).
* **Why Post-Hoc Adjustment and Category-Index Ordering Are Wrong**:
  - If you modify $\beta_c^{\text{hinge}}$ *after* training (post-hoc), the other blocks ($\beta_{c,p}^{\text{item}}$ and $\beta_c^{\text{count}}$) have already adapted to the old value during optimization, throwing the joint model out of equilibrium.
  - If you shrink price parameters toward a line over category index $c = 0, 1, \dots, C-1$, your model depends on how the categories happen to be numbered in the database! Category numbers are arbitrary labels with no economic ordering.

**Our Solution (`regularization_loss` + `fit_with_validation_selection` in [`proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py))**:
Inspired by Empirical-Bayes hierarchical pooling ([Efron & Morris 1973](https://www.tandfonline.com/doi/abs/10.1080/01621459.1973.10481350)), we place a **strictly permutation-invariant hierarchical quadratic penalty** directly inside the L-BFGS-B training objective:

$$\mathcal{R}_{\text{price}}(\theta) = \sum_{c=1}^C \sum_{p=1}^P m_{c,p}^{\text{item}} (\beta_{c,p}^{\text{item}})^2 + \sum_{b \in \{\text{cat}, \text{hinge}, \text{count}, \text{cross}\}} w_b \sum_{c=1}^C m_{b,c} (\beta_{b,c})^2 + \bar{m}^{\text{item}}\left[ w_{\text{within}} \sum_{c=1}^C \sum_{p=1}^P (\beta_{c,p}^{\text{item}} - \bar{\beta}_{c,\cdot}^{\text{item}})^2 + w_{\text{between}} P \sum_{c=1}^C (\bar{\beta}_{c,\cdot}^{\text{item}} - \bar{\beta}_{\cdot,\cdot}^{\text{item}})^2 \right] + \bar{m}^{\text{cat}} \sum_{b \in \{\text{cat}, \text{hinge}, \text{count}\}} w_{\text{pool}, b} \sum_{c=1}^C (\beta_{b,c} - \bar{\beta}_{b,\cdot})^2.$$

Why does this work so well?
1. **All 5 price blocks adapt together during L-BFGS-B optimization**: When $w_{\text{hinge}}$ and $w_{\text{pool}}$ regularize noisy hinge fluctuations during training, $\beta_c^{\text{cat}}$, $\beta_c^{\text{count}}$, and $\beta_{c,p}^{\text{item}}$ simultaneously adjust to maximize the joint log-likelihood.
2. **100% Permutation-Invariant**: Every sum is symmetric over categories $c \in \{1, \dots, C\}$ and products $p \in \{1, \dots, P\}$. Permuting category indices in any order produces the **exact same penalty and fitted predictions** (verified to $< 10^{-12}$ in `test_category_permutation_invariance_of_regularization` in [`tests/test_whole_law.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/tests/test_whole_law.py)).
3. **Validation Selection**: `fit_with_validation_selection` fits candidate penalty tiers (`unregularized`, `mild_hierarchical`, `moderate_hierarchical`, `strong_hierarchical`) to certified stationarity and selects the tier with the highest validation log-likelihood. On `missions_b`, this cuts same-category cross-price relative MAE from `85.8%` (and `37.3%` unregularized) down to **`20.7%`** and raises optimal promotion choices to **`5/6`** (`6/6` on `missions_a` and `zero_price`).

---

## Chapter 5: Architecture II — `StructuredEnergy` and `CompositeWholeLawEnergy` for Real Scanner Panels

On real supermarket scanner panels—**ERIM** ($C=7, P=4, J=28$ products, `47,858` training baskets) and **Dunnhumby Complete Journey** ($C=16, P=4, J=64$ products, `87,503` training baskets)—baskets exhibit both **specific product-pair complementarities** (e.g., hot dogs + hot dog buns) and **global basket-size & multi-category mission structure**.

### 5.1 Step 1: `StructuredEnergy` (Category-Frontier Dynamic Programming)
First, how do we build an exact Markov Random Field (energy model) over $J = 64$ products when $2^{64} - 1 \approx 1.84 \times 10^{19}$?
`StructuredEnergy` ([`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) assigns each basket $y \in \{0,1\}^J \setminus \{\mathbf{0}\}$ the energy:

$$\mathcal{E}_{\text{base}}(y \mid z) = \sum_{j=1}^J u_j(z) y_j + \sum_{c=1}^C \gamma_c \frac{n_c(y)(n_c(y) - 1)}{2} + \gamma_{\text{global}} \frac{N(y)(N(y) - 1)}{2} + \sum_{(i, j) \in \mathcal{E}} J_{ij} y_i y_j,$$

where $\mathcal{E}$ is a sparse set of screened pairwise product links (at most 8 edges with bounded frontier width $\le 2$, selected on training data via convex conditional-logit screening in `data/{dataset}_ref/screen.json`).

**How does `StructuredEnergy.partition` sum $\exp(\mathcal{E}_{\text{base}}(y \mid z))$ over all $2^{64} - 1$ baskets in milliseconds?**
1. Within each category $c \in \{1, \dots, C\}$, products that do not touch any edge in $\mathcal{E}$ are called **free products** (`free`). Their subset sums for any count `extra` are computed in $O(P^2)$ using `log_esp`!
2. Products that touch an edge in $\mathcal{E}$ are called **interface products** (`interface`, at most 2 per category).
3. As the dynamic program sweeps across categories $c = 0, 1, \dots, C-1$, it only needs to remember two things in its state tuple `(boundary_mask, running_count)`:
   - the binary state of active edge endpoints that cross into future categories $c' > c$ (`boundary_mask`, at most $2^2 = 4$ states), and
   - the running total number of items bought so far (`running_count` $\in \{0, \dots, (c+1)P\}$).
4. At category $C - 1$, `partition(..., return_final=True)` outputs the exact log-sum of weights $W_{\text{base}}(N \mid z)$ for every total basket size $N \in \{1, \dots, J\}$!

### 5.2 Why `StructuredEnergy` Alone Is Not Enough on Real Data
Look at the Dunnhumby benchmark table ([`results/dunnhumby_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/dunnhumby_comparison.csv)): `StructuredEnergy` gets a test log-likelihood of `-6.94702`, but its basket-size Total Variation error is **`0.08139`** (8.14%). Why?
1. **A single parabola $\gamma_{\text{global}} N(N-1)/2$ cannot fit real supermarket basket-size histograms**: Real grocery trips have a huge spike at $N = 1$ (quick 1-item fill-in trips) and a heavy right tail ($N \ge 10$ weekly stock-up trips). A quadratic function of $N$ is too stiff to match both simultaneously.
2. **Sparse pairwise links $\mathcal{E}$ miss store-wide shopping missions**: Buying breakfast items across 4 categories or cleaning supplies across 5 categories is driven by a latent trip mission, not just 8 pairwise product edges.

### 5.3 Step 2: `CompositeWholeLawEnergy` (The Complete Real-Data Whole-Law Model)
`CompositeWholeLawEnergy` ([`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py)) resolves both limitations while preserving exact $2^J - 1$ normalization and fast backward-DP sampling. It is a mixture of $K_{\text{tot}} = 1 + K_c + K_p$ exact-normalized whole-basket sub-laws:

$$\log P(y \mid z) = \operatorname{logsumexp}_{k=0}^{K_{\text{tot}}-1} \Big( \log \pi_k(h) + \log P_k(y \mid z) \Big).$$

Let's examine each component in plain words:

#### 1. Price-Invariant Pre-Trip Mission Prior (`allow_price_mix=False`)
The mixture weights $\pi(h) = \operatorname{softmax}(\alpha^{\text{mix}} + H_h^{\text{mix}})$ depend **only on the household $h$**, never on shelf prices $d_z$. Why? Because a shopper decides whether they need a quick 1-item trip or a weekly stock-up trip *before* walking into the store and seeing shelf price tags.

#### 2. Component $0$ ($P_0$): Stratified Category-Frontier DP Energy with Nonparametric Count Law $g(N)$ and Quick-Trip Stratum $W_1$
Component 0 takes the exact un-quadratized size-$N$ partition weights $W_{\text{base}}(N \mid z)$ from `StructuredEnergy` and replaces the rigid quadratic $\gamma_{\text{global}} N(N-1)/2$ with three flexible, exact components:
* **Dedicated Size-1 Quick-Trip Stratum ($N = 1$)**: Single-item baskets have their own utility shift $\delta_1$ and rank-8 cross-price bridge $(d R_1) L_1^\top$:
  $$u_j^{(1)}(z) = u_j(z) + \delta_{1,j} + H_{h,j}^{(1)} + \big((d_z R_1) L_1^\top\big)_j - d_{z,j} (L_1 \odot R_1)_j, \qquad W_1(z) = \operatorname{logsumexp}_{j=1}^J u_j^{(1)}(z).$$
* **Nonparametric Global Basket-Size Potential $g(N)$ for $N \ge 2$**: Instead of a parabola, `dg` assigns an unconstrained parameter $g(N)$ to every basket size $N \in \{2, \dots, 17\}$ (with a monotone decay slope $-\operatorname{softplus}(\text{tail\_slope})\cdot(N - 17)$ for $N > 17$).
* **Household Count Slope $s_h (N - 1)$**: Allows each household $h$ to prefer systematically larger or smaller baskets.

Thus, the exact log-probability of basket $y$ (with size $N = |y|$) under Component 0 is:

$$\log P_0(y \mid z) = \begin{cases} u_{j^*}^{(1)}(z) - \log Z_0(z) & \text{if } N(y) = 1 \text{ (where } y_{j^*} = 1\text{)}, \\[6pt] \mathcal{E}_{\text{unquad}}(y \mid z) + g(N) + s_h(N - 1) - \log Z_0(z) & \text{if } N(y) \ge 2, \end{cases}$$

where $\log Z_0(z) = \operatorname{logsumexp}\!\big(\{W_1(z)\} \cup \{W_{\text{base}}(N \mid z) + g(N) + s_h(N-1)\}_{N=2}^J\big)$.

#### 3. Category Mission Components ($P_k^{\text{cat}}, k = 1 \dots K_c$)
To capture multi-category trip archetypes (initialized from the top singular vectors of the empirical category-incidence matrix on training data), each category $c$ has an inclusive nonempty log-weight $\log W_c(u_c)$ computed in $O(P^2)$ via `_category_inclusive_weights`, shifted by category mission prototype $M_{k,c}^{\text{cat}}$ and category price/household shift $\Delta_c(z) = \kappa_c^{\text{shift}} + H_{h,c}^{\text{cat}} + (d_z \Phi^{\text{cat}})_c$. Conditioned on $y \neq \mathbf{0}$, its exact partition function is:

$$\log Z_k^{\text{cat}}(z) = A_k^{\text{cat}}(z) + \log\!\big(1 - \exp(-A_k^{\text{cat}}(z))\big), \qquad A_k^{\text{cat}}(z) = \sum_{c=1}^C \operatorname{softplus}\!\big(\log W_c(u_c) + \Delta_c(z) + M_{k,c}^{\text{cat}}\big).$$

#### 4. Product Bundle Mission Components ($P_m^{\text{prod}}, m = 1 \dots K_p$)
Initialized from the top singular vectors of the product co-occurrence matrix on training data, each product mission $m$ shifts item utilities by a fixed bundle prototype $M_m^{\text{prod}} \in \mathbb{R}^J$ and category shift $M_{m,c}^{\text{prod-cat}} \in \mathbb{R}^C$, normalized in closed form via `_category_inclusive_weights` in $O(C P^2)$.

When `CompositeWholeLawEnergy` is added on top of `StructuredEnergy` on Dunnhumby ([`results/dunnhumby_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/dunnhumby_comparison.csv)), test log-likelihood improves by **`+0.1130` nats per basket** (`-6.94702` $\to$ **`-6.83402`**, with a 95% household-bootstrap CI of `[+0.1041, +0.1224]`), and Basket-Size Total Variation drops by **nearly $4\times$** (`0.08139` $\to$ **`0.02166`**)!

---

## Chapter 6: All 11 Baseline Models Explained Simply

To test the Whole-Law Basket Model rigorously, [`whole_law_basket/baselines.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/baselines.py) implements **11 distinct baseline models** spanning six modeling paradigms. Every single baseline is normalized over the exact same nonempty basket space $\mathcal{S}_+ = \{0,1\}^J \setminus \{\mathbf{0}\}$ and provides an exact `.sample(ctx, seed)` method:

### Paradigm A: Independent Binary Logit Models
1. **`popularity` (`PooledPopularity`)**:
   * *Plain words*: Every product $j$ has a single store-wide popularity intercept $\lambda_j$, ignoring who the shopper is and what the prices are.
   * *Formula*: $\log P(y \mid z) = \sum_{j=1}^J \lambda_j y_j - A - \log(1 - e^{-A})$, where $A = \sum_{j=1}^J \operatorname{softplus}(\lambda_j)$.
2. **`no_price` (`NoPriceIndependent`)**:
   * *Plain words*: Each shopper has personal item preferences, store effects, and display/mailer lifts, but price sensitivity is locked at zero ($\beta_j = 0$). Measures how much predictive and behavioral power comes purely from price variation.
3. **`independent_parent` (`PersonalizedIndependent` / MVP Logit)**:
   * *Plain words*: Full personalized independent binary logit with item-specific price sensitivities $\beta_j d_j$, conditioned on buying at least one item.

### Paradigm B: Elementary Symmetric Polynomial (ESP) Count & Category Models
4. **`size_aware_esp` (`SizeAwareESP`)**:
   * *Plain words*: First predicts how many total items $N \in \{1, \dots, J\}$ the shopper will buy using a softmax over basket sizes (tilted by household need and store-average price), and then picks $N$ items across the whole store using an ESP subset draw $\propto \exp(\sum_j u_j y_j) / E_N(e^u)$.
5. **`independent_categories_esp` (`IndependentCategoriesESP`)**:
   * *Plain words*: Treats each category $c$ as an independent module with its own category-visit logit $\eta_c = \kappa_c + H_{h,c} + \operatorname{softplus}(\tau_c)\,\text{IV}_c$ and within-category quadratic count penalty $\rho_c n_c(n_c-1)/2 - \nu_c(n_c-1)$, conditioned on visiting at least one category.
6. **`mission_esp` (`LatentMissionESP`)**:
   * *Plain words*: A $K = 2$ latent-mission mixture of `IndependentCategoriesESP` models. Unlike `CalibratedMissionLaw`, it uses ESP within-category subsets rather than continuous-time Plackett–Luce races and lacks category-level price/cross-price terms.

### Paradigm C: Fixed-Order Autoregressive Binary Models
7. **`pairwise_ar` (`PairwiseBinaryAR`)**:
   * *Plain words*: Numbers products $j = 1, \dots, J$ and models the logit of buying product $j$ as $u_j + \sum_{i < j} W_{j,i} y_i$, where $W$ is a lower-triangular $J \times J$ interaction matrix.
8. **`low_rank_bilinear_ar` (`LowRankBilinearAR`, alias `ShopperFactorAR`)**:
   * *Plain words*: A fixed-order binary autoregression where the prefix interaction matrix $\operatorname{tril}(\alpha \rho^\top, -1)$ is low-rank ($R = 8$ on real data, $R = 6$ on synthetic) plus a running basket-size slope. *(Note: Unlike the unordered sequential choice-and-checkout model of [Ruiz, Athey & Blei 2020](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-1/SHOPPER--A-probabilistic-model-of-consumer-choice-with-substitutes/10.1214/19-AOAS1265.short), which sums over item arrival orderings, `LowRankBilinearAR` evaluates binary conditionals in a fixed product order so exact nonempty normalization $\log(1 - \prod_j \sigma(-u_j))$ holds in closed form.)*

### Paradigm D: Latent Continuous Factor Mixture Models
9. **`gaussian_factor_mixture` (`GaussianFactorMixture`, alias `LogisticNormalVAE`)**:
   * *Plain words*: Assumes each trip has a latent continuous Gaussian taste vector $z_m \in \mathbb{R}^K$ drawn from $M = 12$ quasi-Monte Carlo Gaussian nodes ([Manchanda, Ansari & Gupta 1999](https://pubsonline.informs.org/doi/10.1287/mksc.18.2.95) style random-effects covariance across products via factor loadings $V \in \mathbb{R}^{J \times K}$), normalized exactly over nonempty baskets at every node $m$.

### Paradigm E: Deep Autoregressive Neural Networks
10. **`neural_nade` (`FastNeuralNADE`, [Uria et al. 2016](https://jmlr.org/papers/v17/16-272.html))**:
    * *Plain words*: A conditional Neural Autoregressive Distribution Estimator with hidden width $H = 16$, a rank-8 cross-price bridge $(d R) L^\top$, a price-to-hidden encoding matrix $Q \in \mathbb{R}^{J \times 16}$, an autoregressive prefix weight matrix $E \in \mathbb{R}^{J \times 16}$, and an output projection $O \in \mathbb{R}^{J \times 16}$. At step $j$, the hidden state is $\sigma(d Q + y_{<j} E_{<j})$ and the logit for product $j$ is $u_j + \sigma(d Q + y_{<j} E_{<j}) O_j^\top$.

### Paradigm F: Unstratified Category-Frontier DP Energy
11. **`structured_energy` (`StructuredEnergy`)**:
    * *Plain words*: The pure category-frontier DP energy model with quadratic count penalty $\gamma_{\text{global}} N(N-1)/2$ and screened pairwise links $\mathcal{E}$, without the nonparametric count law $g(N)$, size-1 quick-trip stratum $W_1$, or mission mixture components of `CompositeWholeLawEnergy`.

---

## Chapter 7: Optimization, Curvature Preconditioning, and Stationarity Certificates

Why do we train all 12 models with **preconditioned bound-constrained L-BFGS-B** ([Byrd, Lu, Nocedal & Zhu 1995](https://epubs.siam.org/doi/10.1137/0916069)) in `fit_lbfgsb` ([`whole_law_basket/metrics.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/metrics.py)) rather than noisy first-order stochastic gradient descent (SGD/Adam)?

### 7.1 Why Quasi-Newton L-BFGS-B Beats SGD for Structural Basket Models
In a basket model with household fixed effects $H_{h,j}$, global intercepts $\lambda_j$, and price sensitivities $\beta_j$:
* **Household frequency imbalance**: One household might appear in `50` training trips while another appears in only `2` trips. The gradient with respect to a frequent shopper's parameters is $25\times$ larger than for an infrequent shopper.
* **Price variance imbalance**: Product log-prices $d_{c,p}$ have variance $\approx 0.04$, category-mean log-prices $\bar{d}_c$ have variance $\approx 0.01$, and category hinges $\max(\bar{d}_c, 0)$ have variance $\approx 0.003$, while intercepts $\lambda_j$ multiply $1.0$.

Without preconditioning, the Hessian matrix (curvature) is badly ill-conditioned, causing standard gradient descent to zig-zag for thousands of steps.

### 7.2 Diagonal Curvature Preconditioning (`scales`)
In `fit_lbfgsb` ([`whole_law_basket/metrics.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/metrics.py)), we reparameterize $\theta = s \odot v$, where $v$ are the optimizer coordinates and $s_i > 0$ is the inverse square root of the approximate diagonal curvature of parameter $\theta_i$:
* **For household parameters** ($H_h$, `need`, `weights`, `s_house`) where household $h$ appears in fraction $f_h = N_h / N_{\text{train}}$ of trips:
  $$s_h = \left(0.15 f_h + \frac{\text{ridge}}{N_{\text{train}}}\right)^{-1/2}.$$
* **For price parameters** ($\beta_j$, $\beta_j^-$, $\beta_j^+$) with empirical second moment $m_j = \mathbb{E}[d_j^2]$:
  $$s_j = \left(0.15 m_j + \frac{\text{ridge}}{N_{\text{train}}}\right)^{-1/2}.$$

This makes the diagonal of the preconditioned Hessian approximately $1.0$ across every single parameter block!

### 7.3 Dual Projected-Gradient Stationarity Certificates (`gradient_norm`)
Because price elasticities $\beta_j \in [0, 8]$ and utility parameters $\theta_i \in [-15, 15]$ have box bounds $[l_i, u_i]$, a parameter at its lower bound $\theta_i = l_i$ is optimal even if its gradient $\nabla_{\theta_i} \mathcal{L} > 0$ (pushing $\theta_i$ further downward against the wall $l_i$).

The **projected gradient** (`gradient_norm` in [`whole_law_basket/metrics.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/metrics.py)) zeros out gradient components that point outward into an active boundary wall:

$$g_i^{\text{proj}}(\theta) = \begin{cases} 0 & \text{if } \theta_i \le l_i + 10^{-12} \text{ and } \nabla_{\theta_i}\mathcal{L}(\theta) > 0, \\ 0 & \text{if } \theta_i \ge u_i - 10^{-12} \text{ and } \nabla_{\theta_i}\mathcal{L}(\theta) < 0, \\ \nabla_{\theta_i}\mathcal{L}(\theta) & \text{otherwise.} \end{cases}$$

A fit passes our **dual stationarity certificate** (`stationarity_passed = True`) only if the maximum absolute projected gradient is below $2 \times 10^{-4}$ in **both** raw parameter coordinates and preconditioned optimizer coordinates:

$$\max_i \big|g_i^{\text{proj}}(\theta)\big| \le 2 \times 10^{-4} \quad \text{and} \quad \max_i \big|s_i g_i^{\text{proj}}(\theta)\big| \le 2 \times 10^{-4}.$$

---

## Chapter 8: How We Test Basket Models (Likelihood, Generation, Counterfactuals, and Promotions)

A truly useful basket model must pass three distinct tests ([`whole_law_basket/metrics.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/metrics.py) and [`whole_law_basket/synthetic.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/synthetic.py)):

### 8.1 Test 1: Out-of-Sample Predictive Accuracy
1. **Held-Out Test Log-Likelihood (`test_ll`)**: The average exact log-probability $\frac{1}{B_{\text{test}}} \sum_{b=1}^{B_{\text{test}}} \log P(y_b \mid z_b)$ on future test trips. Higher (less negative) is better.
2. **Paired Household-Clustered Bootstrap 95% Confidence Intervals (`cluster_interval`)**: Because trips by the same household are correlated, we resample **entire households** with replacement $1,000$ times to compute a 95% confidence interval on the log-likelihood gain $\Delta \ell_b = \log P_{\text{proposed}}(y_b \mid z_b) - \log P_{\text{baseline}}(y_b \mid z_b)$.
3. **Recall@5 (`recall5`)** and **MC-Corrected Brier Score (`Brier`)**: Using $M = 48$ simulated basket draws per test context, we compute the Monte Carlo marginal purchase probability $\hat{p}_{b,j} = \frac{1}{M}\sum_{m=1}^M \tilde{y}_{b,m,j}$, measure the fraction of true purchased items captured in the top 5 predicted items (`recall5`), and compute the unbiased Brier score:
   $$\text{Brier}_{\text{MC-corrected}} = \frac{1}{B_{\text{test}} J} \sum_{b=1}^{B_{\text{test}}} \sum_{j=1}^J \left[ (\hat{p}_{b,j} - y_{b,j})^2 - \frac{\hat{p}_{b,j}(1 - \hat{p}_{b,j})}{M - 1} \right].$$
   *(Why subtract $\frac{\hat{p}(1-\hat{p})}{M-1}$? Because averaging $M=48$ finite Monte Carlo draws adds binomial sampling variance $\frac{p(1-p)}{M}$; subtracting its unbiased estimator removes finite-sample Monte Carlo penalty so all models are compared on their true underlying probability calibration.)*

### 8.2 Test 2: Distributional Realism of Generated Baskets (`generation_metrics`)
For `2,048` seeded test contexts, every model generates `48` synthetic baskets (`3 replicas x 16 draws = 98,304` total baskets). We compare the generated baskets against the actual test baskets on four structural margins:
1. **Basket-Size Total Variation (`size_TV`)**: Half the $L_1$ distance between the generated basket-size histogram $\hat{P}(N = k)$ and the true test histogram $P_{\text{obs}}(N = k)$:
   $$\text{size\_TV} = \frac{1}{2} \sum_{k=1}^J \big|\hat{P}(N = k) - P_{\text{obs}}(N = k)\big|.$$
2. **Product Frequency MAE (`frequency_MAE`)**: Mean absolute error in item purchase rates $\mathbb{E}[y_j]$ across all $J$ products.
3. **Category-Pair Co-occurrence MAE (`category_pair_MAE`)**: Mean absolute error in the probability $\mathbb{E}[x_c x_{c'}]$ that two categories $c < c'$ are bought together on the same trip.
4. **Within-Category Multiple-Purchase MAE (`within_category_multiple_MAE`)**: Mean absolute error in the probability $\mathbb{E}[\mathbb{I}(n_c \ge 2)]$ of buying 2 or more items in the same category.

### 8.3 Test 3: Causal Price Response & Promotion Optimization (`response_score` & `global_promotion`)
On the synthetic worlds (`missions_a`, `missions_b`, `zero_price`), we know the exact ground-truth data-generating process (`World` in [`whole_law_basket/synthetic.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/synthetic.py)). We test `18` counterfactual price interventions (single-item 20% discounts, single-item 20% price hikes, category-wide 15% discounts, and cross-category price shifts):
* **Own-Price Relative MAE (`own_price_rel_mae`)**: How accurately the model predicts the change in sales $\Delta \mathbb{E}[y_{c,p}]$ of the discounted product itself.
* **Same-Category Cross-Price Relative MAE (`same_category_rel_mae`)**: How accurately the model predicts cannibalization/expansion $\Delta \mathbb{E}[y_{c,q}]$ on *other* products $q \neq p$ in the same category $c$.
* **Cross-Category Relative MAE (`cross_category_rel_mae`)**: How accurately the model predicts spillover $\Delta \mathbb{E}[y_{c',q}]$ on products in *other* categories $c' \neq c$.
* **Optimal Promotion Choices (`optimal_plans_of_6`)**: Across 6 wholesale cost structures, the retailer chooses which promotional plan $a \in \{0, \dots, 18\}$ maximizes expected profit $\sum_j (\text{price}_j(a) - \text{cost}_j)\,\mathbb{E}[y_j \mid a]$. We count how many of the 6 cost scenarios each model gets right (`optimal_plans_of_6`).

---

## Chapter 9: Empirical Results Across ERIM, Dunnhumby, and Synthetic Worlds

All results below come from canonical fresh fits (`--mode fresh`) on 100% of the training splits (`47,858` ERIM baskets, `87,503` Dunnhumby baskets, and `20,000` synthetic baskets per world) with equal `48`-draw generation budgets and traceable checkpoints in [`results/checkpoints/`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/checkpoints).

### 9.1 Real Scanner Panels: ERIM ($J = 28$) and Dunnhumby ($J = 64$)
Full tables: [`results/erim_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/erim_comparison.csv) and [`results/dunnhumby_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/dunnhumby_comparison.csv).

| Rank (Dunnhumby) | Model | ERIM Test LL (`47,858` train) | ERIM Size TV | Dunnhumby Test LL (`87,503` train) | Dunnhumby Size TV (`48` draws) | Dunnhumby Recall@5 | Dunnhumby Brier |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`proposed_whole_law` (`CompositeWholeLawEnergy`)** | **`-3.36726`** | `0.02296` | **`-6.83402`** | `0.02166` | **`0.47315`** | **`0.02729`** |
| 2 | `neural_nade` (`FastNeuralNADE`) | `-3.41268` | `0.02000` | `-6.84261` | **`0.01660`** | `0.45926` | `0.02738` |
| 3 | `structured_energy` (`StructuredEnergy`) | `-3.37127` | `0.01427` | `-6.94702` | `0.08139` | `0.46345` | `0.02740` |
| 4 | `pairwise_ar` (`PairwiseBinaryAR`) | `-3.38805` | `0.02126` | `-7.04281` | `0.05076` | `0.43073` | `0.02789` |
| 5 | `mission_esp` (`LatentMissionESP`) | `-3.46752` | **`0.00769`** | `-7.06080` | `0.02845` | `0.45667` | `0.02758` |
| 6 | `low_rank_bilinear_ar` (`LowRankBilinearAR`) | `-3.39436` | `0.01885` | `-7.13542` | `0.04316` | `0.43341` | `0.02789` |
| 7 | `independent_categories_esp` (`IndependentCategoriesESP`) | `-3.47651` | `0.02359` | `-7.17003` | `0.10899` | `0.43970` | `0.02773` |
| 8 | `size_aware_esp` (`SizeAwareESP`) | `-3.42627` | `0.01820` | `-7.18617` | `0.02208` | `0.44409` | `0.02789` |
| 9 | `no_price` (`NoPriceIndependent`) | `-3.48562` | `0.01635` | `-7.26764` | `0.13546` | `0.43943` | `0.02786` |
| 10 | `independent_parent` (`PersonalizedIndependent`) | `-3.39628` | `0.02716` | `-7.26834` | `0.13403` | `0.43714` | `0.02787` |
| 11 | `gaussian_factor_mixture` (`GaussianFactorMixture`) | `-3.40342` | `0.02929` | `-7.29229` | `0.09115` | `0.42887` | `0.02797` |
| 12 | `popularity` (`PooledPopularity`) | `-4.29747` | `0.06072` | `-8.15657` | `0.14474` | `0.30174` | `0.03002` |

**Key Takeaways from Real Scanner Data**:
1. **`proposed_whole_law` ranks #1 in held-out test log-likelihood on both ERIM (`-3.36726`) and Dunnhumby (`-6.83402`)**, while also achieving the best Recall@5 (`0.47315`) and best MC-corrected Brier score (`0.02729`) on Dunnhumby.
2. **On ERIM**, `proposed_whole_law` beats `neural_nade` by **`+0.04542` nats/basket** (95% household bootstrap CI `[+0.03971, +0.05162]`) and beats `structured_energy` by **`+0.00401` nats/basket** (95% CI `[+0.00039, +0.00744]`).
3. **On Dunnhumby**, adding the nonparametric count potential $g(N)$, quick-trip stratum $W_1$, and pre-trip mission mixtures in `CompositeWholeLawEnergy` improves over `structured_energy` by **`+0.11301` nats/basket** (95% CI `[+0.10411, +0.12238]`) and slashes basket-size TV error from `8.14%` down to **`2.17%`**, outperforming `neural_nade` by `+0.00859` nats/basket on test LL (`+0.03020` nats/basket on validation LL: `-6.42339` vs. `-6.45359`).

### 9.2 Synthetic Counterfactual & Promotion Worlds (`missions_a`, `missions_b`, `zero_price`)
Full tables: [`results/synthetic_missions_a_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/synthetic_missions_a_comparison.csv), [`results/synthetic_missions_b_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/synthetic_missions_b_comparison.csv), and [`results/synthetic_zero_price_comparison.csv`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/results/synthetic_zero_price_comparison.csv).

| World | Model | Test LL | Validation LL | Size TV | Own-Price Rel MAE | Same-Cat Cross Rel MAE | Cross-Cat Rel MAE | Zero-Price MAE | Optimal Plans (of 6) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`missions_a`** | **`proposed_calibrated_mission_law`** | **`-5.23728`** | **`-5.32692`** | **`0.00212`** | `0.08334` | **`0.18574`** | **`0.31082`** | — | **`6/6`** |
| `missions_a` | `uncalibrated_mission_law` | `-5.23753` | `-5.32709` | `0.00206` | **`0.07947`** | `0.19559` | `0.32115` | — | **`6/6`** |
| `missions_a` | `mission_esp` | `-5.26775` | `-5.35843` | `0.00396` | `0.11881` | `0.75771` | `0.38799` | — | `4/6` |
| `missions_a` | `neural_nade` | `-5.28907` | `-5.37404` | `0.01729` | `0.11873` | `0.62130` | `0.84272` | — | `4/6` |
| **`missions_b`** | **`proposed_calibrated_mission_law`** | **`-5.20347`** | **`-5.14456`** | `0.00187` | **`0.05451`** | **`0.20710`** | **`0.18217`** | — | **`5/6`** |
| `missions_b` | `uncalibrated_mission_law` | `-5.20364` | `-5.14529` | **`0.00173`** | `0.05417` | `0.37337` | `0.26786` | — | **`5/6`** |
| `missions_b` | `mission_esp` | `-5.23646` | `-5.17680` | `0.00830` | `0.12259` | `0.84454` | `0.39889` | — | `4/6` |
| `missions_b` | `neural_nade` | `-5.24307` | `-5.18441` | `0.01609` | `0.11123` | `0.58311` | `0.83923` | — | `4/6` |
| **`zero_price`** | **`proposed_calibrated_mission_law`** | **`-5.17618`** | **`-5.18633`** | **`0.00224`** | — | — | — | `0.00037` | **`6/6`** |
| `zero_price` | `uncalibrated_mission_law` | `-5.17651` | `-5.18636` | `0.00226` | — | — | — | `0.00042` | **`6/6`** |
| `zero_price` | `mission_esp` | `-5.19988` | `-5.21117` | `0.00556` | — | — | — | `0.00026` | `3/6` |
| `zero_price` | `neural_nade` | `-5.21789` | `-5.22694` | `0.01959` | — | — | — | `0.00082` | `3/6` |

**Key Takeaways from Synthetic Counterfactual Worlds**:
1. **Why sequential `neural_nade` and `pairwise_ar` struggle on counterfactuals**: Even when `neural_nade` achieves reasonable held-out log-likelihood, its cross-category price response relative MAE is **`84.3%`** on `missions_a` and **`83.9%`** on `missions_b`, and it fails to identify the profit-maximizing promotion on `zero_price` (`3/6`) and `missions_a`/`missions_b` (`4/6`).
2. **Why `CalibratedMissionLaw` succeeds**: By modeling pre-trip missions, category counts, and unordered Exponential-Race Plackett–Luce choice with joint permutation-equivariant hierarchical price regularization, `proposed_calibrated_mission_law` achieves **`18.6%` same-category relative MAE (`6/6` optimal plans) on `missions_a`**, **`20.7%` same-category relative MAE (`5/6` optimal plans) on `missions_b`**, and **`3.72e-4` zero-price MAE (`6/6` optimal plans) on `zero_price`**.

---

## Chapter 10: Codebase Map and Annotated References

### 10.1 File-by-File Guide to the Repository
* [`whole_law_basket/race_pl.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/race_pl.py): Exact-mode-centered analytic $\sinh$ log-time Gauss–Legendre quadrature (`race_subset_log_probabilities`, `race_observed_log_probability`, `_find_log_s_mode`, and `verify_race_pl_quadrature`).
* [`whole_law_basket/proposed_model.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/proposed_model.py): Core implementation of `log_esp`, `_sample_esp_subset_local`, `StructuredEnergy`, `CompositeWholeLawEnergy`, and `CalibratedMissionLaw` (including `regularization_loss` and `fit_with_validation_selection`).
* [`whole_law_basket/baselines.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/baselines.py): All 10 baseline classes (`PooledPopularity`, `NoPriceIndependent`, `PersonalizedIndependent`, `SizeAwareESP`, `IndependentCategoriesESP`, `LatentMissionESP`, `PairwiseBinaryAR`, `LowRankBilinearAR`, `GaussianFactorMixture`, `FastNeuralNADE`).
* [`whole_law_basket/metrics.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/metrics.py): Preconditioned L-BFGS-B optimizer (`fit_lbfgsb`), projected gradient norm (`gradient_norm`), parallel sampler (`draws`), and evaluation metrics (`generation_metrics`, `predictive_metrics`, `cluster_interval`).
* [`whole_law_basket/real_data.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/real_data.py): Standalone loader for bundled ERIM and Dunnhumby splits under [`data/`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/data).
* [`whole_law_basket/synthetic.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/whole_law_basket/synthetic.py): Ground-truth synthetic mission DGP (`World`), counterfactual price action suite (`actions`, `response_score`), and profit-maximizing promotion evaluator (`global_promotion`).
* [`benchmarks/run_all_benchmarks.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/benchmarks/run_all_benchmarks.py): End-to-end benchmark runner supporting `--mode fresh` (default) and `--mode resume`.
* [`tests/test_whole_law.py`](file:///Users/ajit/Projects/nf_dunnhumby/ebi_chain/whole_law_basket_repo/tests/test_whole_law.py): Automated unit test suite verifying quadrature accuracy, category permutation invariance, exact $2^J - 1$ partition normalization across all models, sampler TV accuracy, and input-price gradients.

### 10.2 Foundational Academic References
1. **Besag, J. (1974)**. [Spatial Interaction and the Statistical Analysis of Lattice Systems](https://academic.oup.com/jrsssb/article/36/2/192/7027394). *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192–225. *(Foundational paper on Markov Random Fields, Ising/auto-logistic potentials, and pseudolikelihood screening used to select our sparse graph edges $\mathcal{E}$.)*
2. **Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995)**. [A Limited Memory Algorithm for Bound Constrained Optimization](https://epubs.siam.org/doi/10.1137/0916069). *SIAM Journal on Scientific Computing*, 16(5), 1190–1208. *(The L-BFGS-B bound-constrained quasi-Newton algorithm and projected-gradient stationarity criterion used in `fit_lbfgsb`.)*
3. **Efron, B., & Morris, C. (1973)**. [Stein's Estimation Rule and Its Competitors—An Empirical Bayes Approach](https://www.tandfonline.com/doi/abs/10.1080/01621459.1973.10481350). *Journal of the American Statistical Association*, 68(341), 117–130. *(Theoretical foundation for hierarchical Empirical-Bayes pooling across exchangeable groups without arbitrary index ordering.)*
4. **Luce, R. D. (1959)**. [Individual Choice Behavior: A Theoretical Analysis](https://www.statisticshowto.com/plackett-luce-model/). New York: Wiley. *(Introduces Luce's Choice Axiom and positive ratio-scale choice weights $w_i$.)*
5. **Manchanda, P., Ansari, A., & Gupta, S. (1999)**. [The "Shopping Basket": A Model for Multicategory Purchase Incidence Decisions](https://pubsonline.informs.org/doi/10.1287/mksc.18.2.95). *Marketing Science*, 18(2), 95–114. *(Seminal marketing science paper on cross-category purchase incidence and unobserved trip/household heterogeneity.)*
6. **McFadden, D. (1974)**. [Conditional Logit Analysis of Qualitative Choice Behavior](https://escholarship.org/uc/item/61s3q2xr). In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105–142). Academic Press. *(Establishes the Thurstonian Gumbel random-utility foundation of logit and Plackett–Luce choice models.)*
7. **Plackett, R. L. (1975)**. [The Analysis of Permutations](https://academic.oup.com/jrsssc/article/24/2/193/6953554). *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 24(2), 193–202. *(Defines the Plackett–Luce distribution over rankings and multi-item draws without replacement.)*
8. **Ruiz, F. J. R., Athey, S., & Blei, D. M. (2020)**. [SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-1/SHOPPER--A-probabilistic-model-of-consumer-choice-with-substitutes/10.1214/19-AOAS1265.short). *Annals of Applied Statistics*, 14(1), 1–27. *(Low-rank bilinear interaction representation for market baskets.)*
9. **Russell, G. J., & Petersen, A. (2000)**. [Analysis of Cross Category Dependence in Market Basket Selection](https://www.researchgate.net/publication/343484851_Market_Basket_Analysis_Recommendation_System_Using_Association_Rules). *Journal of Retailing*, 76(3), 367–392. *(Auto-logistic conditional modeling of cross-category market basket dependence.)*
10. **Takahasi, H., & Mori, M. (1974)**. [Double Exponential Formulas for Numerical Integration](https://ems.press/journals/prims/articles/2686). *Publications of the Research Institute for Mathematical Sciences*, 9(3), 721–741. *(Foundation for exponential/hyperbolic-sine $\sinh$ coordinate transformations in numerical quadrature.)*
11. **Uria, B., Côté, M.-A., Gregor, K., Murray, I., & Larochelle, H. (2016)**. [Neural Autoregressive Distribution Estimation](https://jmlr.org/papers/v17/16-272.html). *Journal of Machine Learning Research*, 17(205), 1–37. *(The NADE architecture underlying our `FastNeuralNADE` baseline.)*
