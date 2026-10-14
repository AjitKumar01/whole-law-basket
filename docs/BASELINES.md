# Basket baselines, from the ground up

This guide explains the baselines **implemented in this repository**, rather than idealized versions of their model families. Each step introduces a new way to explain a basket. Equations define the model; the paragraphs explain why each term is there.

Start with Sections 1–3. Sections 4–14 cover all 11 baseline families. Section 15 shows what is actually trained, and Section 16 explains how to compare and run them. For the proposed models, see [THEORY.md](THEORY.md).

## 1. What are we trying to predict?

Suppose a shop has bread, milk, cereal, and coffee. A shopper buys bread and milk. We represent that basket as:

$$y=(1,1,0,0).$$

A one means the product appears; a zero means it does not. Buying three bottles of milk still gives milk a value of one. Our current prediction target is **distinct-product presence**, not quantities.

The model receives a context $x$: household identity, offered prices, and any prepared preference controls. It assigns a probability to each possible nonempty basket:

$$P(y\mid x),\qquad y\in\{0,1\}^{J}\setminus\{0\}.$$

All basket probabilities must add to one. A model must therefore explain both why particular products appear and why the basket contains that many products.

We use the following notation throughout:

| Symbol | Meaning |
|---|---|
| $J$ | Number of products |
| $C$ and $P$ | Number of categories and products per category; here $J=CP$ |
| $h$ and $H$ | Current household and number of household IDs |
| $y_j$ | Whether product $j$ is in the basket |
| $n_c=\sum_{j\in c}y_j$ | Number of distinct products bought in category $c$ |
| $N=\sum_jy_j$ | Total distinct basket size |
| $d_j=\log(p_j/p_j^{\mathrm{ref}})$ | Offered log price relative to a fixed reference price |
| $b_j(x)$ | Supplied price-free preference score; includes prepared household/store/calendar/display/mailer effects on real data |
| $u_j(x)$ | Product score after adding the model's intercept, preference adjustment, and price effects |
| $\sigma(t)=1/(1+e^{-t})$ | Converts a score into a probability |

A score is not itself a purchase probability. For a basic independent model, $\sigma(u_j)$ is the probability before removing the empty basket. Other models also account for counts, interactions, or hidden states.

A 10% discount adds $\log(0.9)$ to the affected product's $d_j$. To ask a price what-if question, change those prices while holding the other context fields fixed, then recompute the complete distribution or sample from it. The resulting change is a **model prediction**, not automatically a causal fact.

### The shared real-data controls

Most real-data baselines receive the same prepared $b_j(x)$. Thus even a simple residual model can already use rich household preferences. `popularity` is the exception: it ignores these controls.

These shared controls were fitted previously and are bundled separately. Their preparation cost must not be counted as zero when discussing the cost of the complete pipeline. The term `independent_parent` also names a baseline fitted by the runner; it is not the same object as the bundled real-data preference-control model.

## 2. Why exclude the empty basket?

The benchmark describes trips containing at least one item from the selected catalogue. For a probability law $P_0$ initially defined over all binary baskets, conditioning gives:

$$P_+(y\mid x)=\frac{P_0(y\mid x)}{1-P_0(0\mid x)},\qquad y\ne0.$$

This is more than deleting empty generated samples. Both likelihood evaluation and sampling must use the same conditional distribution. The implementations sample nonempty baskets directly.

Conditioning also introduces dependence. With just bread and milk, once we know the basket cannot be empty, the absence of bread makes milk certain. That dependence does not prove an economic substitution relationship.

Every baseline below uses the nonempty support. Models that generate a positive basket size already satisfy it by construction. Mixture models normalize their individual components over nonempty baskets before mixing them.

## 3. What does “increasing complexity” mean?

The order below is a learning path:

| Step | Runner name | New idea introduced |
|---:|---|---|
| 1 | `popularity` | Learn how common each product is |
| 2 | `no_price` | Add shopper/context preferences |
| 3 | `independent_parent` | Add own-product price response |
| 4 | `size_aware_esp` | Explicitly choose basket size |
| 5 | `independent_categories_esp` | Choose categories and counts within them |
| 6 | `mission_esp` | Share a hidden shopping mission across categories |
| 7 | `gaussian_factor_mixture` | Represent hidden trip types using factor-based components |
| 8 | `low_rank_bilinear_ar` | Let earlier product decisions affect later ones using compact factors |
| 9 | `pairwise_ar` / `pairwise_binary_ar` | Give each preceding-product relationship its own coefficient |
| 10 | `neural_nade` | Replace linear prefix effects with a nonlinear hidden representation |
| 11 | `structured_energy` | Score the entire basket jointly and normalize with a graph-based calculation |

This is **not** a strict ranking of speed, parameter count, expressiveness, or accuracy. The models branch: a hidden-mission model and an autoregressive model make different assumptions. A small neural model can be cheaper than a dense pairwise model, while a simple-looking energy equation can have a difficult normalization calculation.

We distinguish three kinds of complexity:

1. **Behavioral structure:** which patterns the model can represent.
2. **Statistical complexity:** how many adjustable quantities the data must estimate.
3. **Computational complexity:** how much work it takes to fit, evaluate, and sample.

Adding structure is useful only when it explains a pattern that simpler models miss.

## 4. Step 1: popularity

**Class:** [`PooledPopularity`](../whole_law_basket/baselines.py).

Start with one score $a_j$ per product. Everyone receives the same scores, at every price:

$$u_j=a_j.$$

The nonempty basket law is:

$$P(y\mid x)=\frac{\exp(\sum_j a_jy_j)}{Z_+(a)},\qquad
Z_+(a)=\prod_{j=1}^{J}(1+e^{a_j})-1.$$

Before nonempty conditioning, each item is an independent yes/no decision with probability $\sigma(a_j)$.

**Example:** bread is popular, so it appears often. The model does not distinguish a household that prefers cereal from one that prefers coffee.

**What this tests:** whether a more elaborate model adds value beyond aggregate product frequencies.

**What it cannot learn:** household differences, price response, product-specific interactions, or latent shopping missions. Any apparent relationship between products comes from conditioning or pooling, not a learned interaction.

**Complexity:** $J$ fitted intercepts. Likelihood evaluation is linear in $J$; the normalizer factorizes instead of enumerating $2^J$ baskets.

## 5. Step 2: no-price personalized independence

**Runner:** `no_price`. **Class:** [`NoPriceIndependent`](../whole_law_basket/baselines.py).

Now allow preferences to differ by shopper or supplied context:

$$u_j(x)=b_j(x)+a_j+A_{h,j}.$$

Use the same nonempty independent law:

$$P(y\mid x)=\frac{\exp(\sum_j u_j(x)y_j)}{\prod_j(1+e^{u_j(x)})-1}.$$

Here $A_{h,j}$ is a household adjustment. In synthetic runs it is learned. In real runs it is frozen at zero because the supplied $b_j(x)$ already contains household preferences.

**Example:** a cereal-loving household receives a larger cereal score than another household. Changing shelf price alone does not change either score.

**Why this is a step beyond popularity:** it separates shopper composition from overall popularity. Otherwise a model might appear to understand baskets simply because it knows which households buy which products.

**Why keep this baseline:** it is the price ablation. Under an intervention that changes only `dlp`, its predicted distribution stays unchanged. It tells us whether price-aware models improve on preference-only predictions.

**Complexity:** up to $HJ$ household adjustments plus $J$ intercepts, but only the $J$ intercepts are fitted in the current real runner. Per-context evaluation remains linear in $J$.

## 6. Step 3: price-aware personalized independence

**Runner:** `independent_parent`. **Class:** [`PersonalizedIndependent`](../whole_law_basket/baselines.py).

Add a direct response to a product's own relative log price:

$$u_j(x)=b_j(x)+a_j+A_{h,j}-\beta_jd_j.$$

The basket law remains:

$$P(y\mid x)=\frac{\exp(\sum_j u_j(x)y_j)}{\prod_j(1+e^{u_j(x)})-1}.$$

A positive $\beta_j$ lowers the product score when its price rises. These own-price coefficients are constrained to be nonnegative by the current fitting bounds. Household adjustments are learned on synthetic data and frozen at zero on real data.

**Example:** discounting milk increases its score and changes how often milk appears.

**What this adds:** actual price sensitivity can now be estimated rather than assumed absent.

**What remains missing:** there is no freely learned bread–milk interaction or cross-price matrix. Prices can still affect other marginal purchase probabilities through the nonempty conditioning, so “independent” does not mean every cross-product marginal response is exactly zero.

**Complexity:** up to $HJ+2J$ fitted quantities; $2J$ in current real runs. Evaluation remains linear in $J$. This is an important low-cost price-aware benchmark.

## 7. Step 4: choose basket size, then choose products

**Runner:** `size_aware_esp`. **Class:** [`SizeAwareESP`](../whole_law_basket/baselines.py).

Independent item decisions imply a basket-size distribution indirectly. This baseline gives size its own model.

First choose $N\in\{1,\ldots,J\}$:

$$q(N=n\mid x)=\operatorname{softmax}_{n=1,\ldots,J}
\left[a_n+(g_h+\xi\bar d)(n-1)\right],
\qquad \bar d=\frac1J\sum_j d_j.$$

The $a_n$ terms set the overall size profile. Household slope $g_h$ tilts it toward smaller or larger baskets. Price slope $\xi$ lets the mean price affect size.

Then choose a product subset of that size:

$$P(y\mid N=n,x)=\frac{\exp(\sum_j u_j(x)y_j)}{E_n(u)},\qquad
E_n(u)=\sum_{S:|S|=n}\exp\left(\sum_{j\in S}u_j\right).$$

The complete law is:

$$P(y\mid x)=q(N(y)\mid x)\,P(y\mid N(y),x).$$

**Example:** first decide this is a three-item trip. Then choose which three items, favoring products with larger scores. Products compete for the three available places, even without pair-specific interaction coefficients.

**What “ESP” means:** $E_n$ is an elementary symmetric polynomial. It sums the weights of all size-$n$ subsets without listing them. With $w_j=e^{u_j}$, the recurrence is:

$$E_{t,n}=E_{t-1,n}+w_tE_{t-1,n-1}.$$

It asks whether product $t$ is absent or present. Computation can be truncated at the largest size needed; computing every size costs $O(J^2)$ rather than enumerating $2^J$ baskets.

**Implementation detail:** product-choice scores are fixed during this baseline's fit. Synthetic runs use the fitted independent parent's utilities; real runs use the supplied preferences and fixed initial price slopes. Only the size distribution is newly estimated. Frozen within-size likelihood terms can therefore be cached.

**Complexity:** $J+H+1$ new fitted parameters, plus the cost of its inputs or parent. It adds a whole-basket count model and a subset-normalization calculation, not a table of product-pair relationships.

## 8. Step 5: choose categories, counts, and products

**Runner:** `independent_categories_esp`. **Class:** [`IndependentCategoriesESP`](../whole_law_basket/baselines.py).

A single basket size does not distinguish “three drinks” from “one drink, one bread, and one dairy product.” This model uses the category hierarchy.

For each category $c$, define the positive-count log weights:

$$v_c(n)=\log E_{c,n}(u)-\rho_c\binom n2+\nu_c(n-1),\qquad n=1,\ldots,P.$$

Here $E_{c,n}$ sums the weights of size-$n$ subsets **within category $c$**. The coefficients $\rho_c$ and $\nu_c$ reshape its count distribution. Their signs are learned; the curvature can penalize or favor multiple purchases.

Normalize these weights and summarize category attractiveness:

$$q_c(n\mid x)=\operatorname{softmax}_n v_c(n),\qquad
I_c(x)=\log\sum_{n=1}^{P}e^{v_c(n)}.$$

Next compute category participation scores:

$$\eta_c(x)=\kappa_c+B_{h,c}+\operatorname{softplus}(\tau_c)I_c(x).$$

The term $I_c$ is an inclusive value: it measures how attractive the available product combinations are. Discounts can change it. The softplus makes its coefficient positive.

Let $z_c=1[n_c>0]$. Category participation follows independent Bernoulli gates, conditioned on at least one active category:

$$P(z\mid x)=\frac{\prod_c\sigma(\eta_c)^{z_c}[1-\sigma(\eta_c)]^{1-z_c}}
{1-\prod_c[1-\sigma(\eta_c)]}.$$

For every active category, choose its count and then its products:

$$P(y\mid x)=P(z(y)\mid x)
\prod_{c:z_c=1}\left[q_c(n_c\mid x)
\frac{\exp(\sum_{j\in c}u_jy_j)}{E_{c,n_c}(u)}\right].$$

**Example:** decide whether to visit dairy; if yes, choose one or two dairy products and then select them. This separates entering a category from choosing a product inside it.

**What this adds:** category participation, multiple purchases within a category, and a route from product attractiveness to category demand.

**What it still lacks:** direct learned relationships between different category gates, beyond nonempty conditioning. Within-category selection at a fixed count uses product scores, not pair-specific bundle bonuses.

**Implementation detail:** base item utilities are fixed during this fit. The category household adjustments $B_{h,c}$ are fitted on synthetic data and frozen at zero on real data. The `item_lam` array is also frozen.

**Complexity:** $4C+HC$ new parameters on synthetic data and $4C$ on real data. Category ESP work scales as $O(CP^2)$ for all counts; fixed terms can be cached. This illustrates why using a hierarchy can reduce computation compared with one catalogue-wide subset calculation.

## 9. Step 6: introduce a shared shopping mission

**Runner:** `mission_esp`. **Class:** [`LatentMissionESP`](../whole_law_basket/baselines.py).

Categories can co-occur because of an unobserved trip purpose. A breakfast trip might raise demand for dairy, cereal, and coffee together.

Introduce a mission $m\in\{1,\ldots,K\}$, with household-specific probabilities:

$$\pi_m(h)=\operatorname{softmax}_m(w_{h,m}),\qquad K=2\text{ in the runner}.$$

Each mission has its own version of the previous category/count law, denoted $P_m(y\mid x)$. Its product scores and gate are:

$$u_{m,c,j}=u^{\mathrm{base}}_{c,j}+t_{m,c,j},$$

$$\eta_{m,c}=\kappa_{m,c}+\operatorname{softplus}(\tau_{m,c})I_{m,c}.$$

The count parameters $\rho_{m,c},\nu_{m,c}$ also vary by mission. Mix the component distributions:

$$P(y\mid x)=\sum_{m=1}^{K}\pi_m(h)P_m(y\mid x).$$

**How generation works:** draw a mission, draw active categories, draw counts, and draw products. The mission is not observed in the training data; fitting uses the summed mixture probability.

**Why this adds behavior:** categories share a hidden cause. They can therefore move together even when their gates are independent within a mission. The mission prior depends on household identity, not prices; conditional basket probabilities still respond to price.

**Important restriction in this code:** the small mission-specific product-taste offsets $t$ are initialized from a seeded random draw and kept fixed. Category effects, count parameters, inclusive-value slopes, and household mixture weights are learned. This is a restricted two-mission baseline, not an unrestricted mission model.

**Complexity:** $4KC+HK$ newly fitted parameters plus fixed taste offsets and any parent. Evaluating $K$ components roughly multiplies the category-law work by $K$. Interpreting a fitted component as “breakfast” requires evidence; a mixture component has no built-in semantic label.

## 10. Step 7: a factor-based mixture of trip types

**Runner:** `gaussian_factor_mixture`. **Class:** [`GaussianFactorMixture`](../whole_law_basket/baselines.py).

Rather than assigning a separate product preference to every trip type, describe trip types with shorter factor vectors. For example, several products might share a factor that increases their scores together.

The code creates $M=12$ fixed vectors $z_m\in\mathbb R^F$ from a seeded Gaussian draw; the first vector is set to zero. The runner uses $F=4$ on synthetic data and $F=6$ on real data.

For component $m$:

$$u_{m,j}(x)=u_j^{\mathrm{base}}(x)+a_j+V_j^\top z_m.$$

Normalize an independent basket law within each component and mix it:

$$P_m(y\mid x)=\frac{\exp(\sum_j u_{m,j}y_j)}{\prod_j(1+e^{u_{m,j}})-1},$$

$$\pi_m(h)=\operatorname{softmax}_m(v_m+W_{h,m}),\qquad
P(y\mid x)=\sum_{m=1}^{M}\pi_m(h)P_m(y\mid x).$$

**Example:** one component raises dairy and cereal together; another raises coffee. A household can use both trip types with different probabilities.

**Why it appears later in the guide:** it adds multiple latent dimensions and component-specific score shifts. Unlike `mission_esp`, it does not model category counts explicitly. It is a different branch of complexity, not a strict extension of that baseline.

**What is actually learned:** on synthetic data, intercepts $a$, loadings $V$, and mixture weights are fitted on top of a frozen parent. On real data, $a$ and $V$ are frozen; only global and household component weights are fitted. Nodes, node scales, and this class's price vector are fixed in both cases. The real variant therefore selects among fixed component patterns rather than learning a general factor representation.

**Complexity:** component evaluation is approximately $O(MJ)$ per context after shifts are formed. Synthetic fitted parameters total $J+JF+HM+M$; real fitted parameters total $HM+M$. Increasing household count can dominate storage even with few factors. Frozen component likelihoods can be cached during real-data fitting.

**Naming:** this is a finite mixture based on Gaussian-drawn nodes. It is not exact integration over a continuous Gaussian, and it has no encoder or variational autoencoder objective. The legacy alias `LogisticNormalFactorMixture` names the same class, not another baseline.

## 11. Step 8: compact interactions between product decisions

**Runner:** `low_rank_bilinear_ar`. **Class:** [`LowRankBilinearAR`](../whole_law_basket/baselines.py).

So far, dependence comes from counts or hidden trip states. Now let one product decision directly change a later product's probability.

Choose a fixed catalogue order. For product $j$, use only earlier decisions $y_{<j}$:

$$\eta_j=u_j(x)+\sum_{i<j}\langle A_j,R_i\rangle y_i+s_j\sum_{i<j}y_i,$$

$$P_0(y\mid x)=\prod_{j=1}^{J}
\sigma(\eta_j)^{y_j}[1-\sigma(\eta_j)]^{1-y_j}.$$

The dot product is a compact interaction coefficient. The final term says that an already-large basket may change the chance of adding product $j$. Normalize away the empty basket:

$$P(y\mid x)=\frac{P_0(y\mid x)}{1-P_0(0\mid x)}.$$

When every preceding decision is zero, all interaction/count terms vanish, so:

$$P_0(0\mid x)=\prod_j[1-\sigma(u_j(x))].$$

**Example:** selecting cereal earlier in the catalogue may raise the later milk score. Similar products share interaction patterns through the factors.

**What is actually learned:** $A$ is a fixed seeded factor matrix; $R$, item intercepts, and size slopes $s$ are fitted. Household factors and own-price coefficients in this class are frozen under both current runner paths. Rank is 6 on synthetic data and 8 on real data.

**A price detail that matters:** the synthetic branch inherits the parent's item intercepts and household preferences, but uses this class's fixed own-price vector, initialized to ones. It does not reuse the parent's fitted price vector. Real runs use the supplied fixed price vector. Thus its price flexibility differs from a fully trainable low-rank AR family.

**Complexity:** the new fitted parameter count is $JR+2J$. It uses fewer interaction parameters than a dense pairwise model. However, the current implementation constructs the dense masked matrix $\operatorname{tril}(AR^\top,-1)$ and performs dense prefix calculations, so runtime/storage do not automatically become linear in $J$ just because the parameters are factorized. The masked matrix need not itself have rank at most $R$.

**Interpretation limit:** catalogue order is a modeling convention, not observed purchase chronology. The interaction coefficients are directional; they are not automatically symmetric economic complements. `ShopperFactorAR` is a legacy alias for this class. It is not a reproduction of SHOPPER.

## 12. Step 9: a separate coefficient for every preceding product

**Runner:** `pairwise_ar` on real data, `pairwise_binary_ar` on synthetic data. Both instantiate [`PairwiseBinaryAR`](../whole_law_basket/baselines.py).

Replace the compact factor dot product with an unrestricted preceding-product coefficient:

$$\eta_j=u_j(x)+\sum_{i<j}W_{ji}y_i.$$

The distribution is the same fixed-order autoregressive construction:

$$P(y\mid x)=\frac{
\prod_j\sigma(\eta_j)^{y_j}[1-\sigma(\eta_j)]^{1-y_j}}
{1-\prod_j[1-\sigma(u_j(x))]}.$$

**Example:** cereal can raise milk's score while coffee has a different effect, without requiring those relationships to share a short factor representation.

**What this adds:** more freedom for individual pair relationships. It does not include the low-rank baseline's explicit running-count slope, so the two models are not simply nested versions of each other.

**What is actually learned:** product intercepts and the dense $W$ array. The likelihood uses only its strict lower triangle. The allocated upper triangle and diagonal do not add predictive interactions, although they remain trainable entries subject to the fitting penalty. Own-price and household parameters are frozen under the current runner. Synthetic parent preferences and fixed unit price slopes are handled as in the low-rank baseline.

The class also contains frozen `cross_left` and `cross_right` arrays. They are not an active cross-price term in this implementation's logits. Do not describe this baseline as learning a separate cross-price bridge.

**Complexity:** $J^2+J$ allocated trainable parameters, of which only $J(J-1)/2$ are useful pair-interaction coefficients. Dense scoring/storage is quadratic in catalogue size. Conditional likelihoods avoid a general whole-basket partition sum; generation still follows the product order.

**Interpretation limit:** a learned pair effect is conditional on earlier decisions. Price changes can propagate through those decisions, but the coefficients alone do not prove causal complementarity.

## 13. Step 10: nonlinear autoregressive interactions

**Runner:** `neural_nade`. **Class:** [`FastNeuralNADE`](../whole_law_basket/baselines.py).

Linear pairwise logits add individual effects. A neural hidden representation can make the effect of a product depend on other selected products and on prices.

First form a price-aware unary score:

$$u_j(x)=b_j(x)+a_j+A_{h,j}-\beta_jd_j
+\sum_{k\ne j}\langle L_j,R_k\rangle d_k.$$

The last term is a rank-8 cross-price bridge. It explicitly excludes its diagonal contribution. Prices also enter the hidden representation:

$$g_j(y_{<j},d)=\sigma\left(\sum_{i<j}y_iE_i+\sum_k d_kQ_k\right),$$

$$\eta_j=u_j(x)+O_j^\top g_j(y_{<j},d).$$

The runner uses hidden width $D=16$. Apply the autoregressive product and nonempty conditioning:

$$P(y\mid x)=\frac{
\prod_j\sigma(\eta_j)^{y_j}[1-\sigma(\eta_j)]^{1-y_j}}
{1-P_0(0\mid x)}.$$

For the empty prefix, the basket contributions disappear but the price-hidden term remains:

$$P_0(0\mid x)=\prod_j\left[1-\sigma\left(u_j(x)+O_j^\top\sigma\left(\sum_kd_kQ_k\right)\right)\right].$$

**Example:** cereal's effect on milk may differ depending on whether coffee is already present. Hidden units combine the prefix into features instead of assigning only one additive coefficient per pair.

**What is actually learned:** item intercepts, own-price slopes, the hidden matrices $E,Q,O$, and both rank-8 bridge matrices. Household adjustments are fitted on synthetic data and frozen at zero on real data, where $b_j(x)$ supplies personalization.

**Why it can be fast despite being “neural”:** this is a small feed-forward autoregressive network, not a large language model or transformer. Prefix hidden sums can be computed with cumulative sums. Work is roughly $O(JD+Jr)$ per context with bridge rank $r=8$, and generation updates a small hidden vector as it visits products. Likelihood evaluation needs no general graph partition function.

**Complexity:** $2J+3JD+2Jr$ active parameters on real data, or $66J$ with these dimensions; synthetic runs add $HJ$. Nonlinearity adds behavioral flexibility without allocating a separate unrestricted matrix for every possible higher-order interaction. It does not guarantee accurate interventions: hidden price paths and interaction terms must be evaluated as a complete model.

## 14. Step 11: score the whole basket jointly

**Runner:** `structured_energy`. **Class:** [`StructuredEnergy`](../whole_law_basket/proposed_model.py). This baseline is used on ERIM and Dunnhumby, not by the current synthetic runner.

Autoregressive models describe ordered yes/no decisions. An energy model instead assigns a single score to the entire basket:

$$F(y,x)=\sum_j u_j(x)y_j
+\sum_c\alpha_c\binom{n_c}2
+\gamma\binom N2
+\sum_{(i,j)\in\mathcal E}\omega_{ij}y_iy_j.$$

The terms have simple roles:

- $u_jy_j$: how attractive each selected product is.
- $\alpha_c\binom{n_c}2$: category-specific preference for more or fewer within-category pairs.
- $\gamma\binom N2$: global basket-size curvature.
- $\omega_{ij}y_iy_j$: a bonus or penalty for an explicitly selected product pair.

Positive count/pair coefficients reward the corresponding combinations; negative coefficients discourage them. Turn scores into probabilities:

$$P(y\mid x)=\frac{e^{F(y,x)}}{Z_+(x)},\qquad
Z_+(x)=\sum_{y'\ne0}e^{F(y',x)}.$$

The unary score has separate slopes below and above the reference price:

$$u_j(x)=b_j(x)+a_j+A_{h,j}
-\beta_j^-\min(d_j,0)-\beta_j^+\max(d_j,0).$$

**Example:** a pair bonus can favor bread and milk together regardless of which is earlier in the product array. The global count term changes all larger baskets, rather than only later conditional logits.

**Why normalization is more demanding:** the score is simple, but $Z_+$ appears to require summing $2^J-1$ baskets. The implementation uses category-frontier dynamic programming plus ESP sums for products without explicit graph links. It tracks counts and the linked-product decisions still needed by later transitions.

Graph structure is critical. If $w$ linked decisions must remain jointly represented, the boundary-state count can scale like $2^w$ times a count dimension, before local-transition and batch factors. Thus parameter count alone does not predict runtime.

**Actual graph in the bundled inputs:** ERIM has **zero** selected explicit product-pair edges. Dunnhumby has **two**, both within categories: `[4,5]` and `[0,3]`. ERIM still has category and global count interactions; neither benchmark configuration is a dense arbitrary product-interaction graph.

**What is actually learned:** item intercepts, two price slopes per product, category count coefficients, one global count coefficient, and selected link strengths. Household adjustments are frozen at zero in the real runner. The shared preference controls remain fixed.

**Complexity:** $3J+C+1+|\mathcal E|$ newly fitted parameters under the current real configuration, but inference cost depends on graph topology and count states. It is a joint-model alternative, not a claim of universally greater flexibility or accuracy than neural NADE. The real proposed composite later uses a fitted instance of this baseline as its frozen base; the baseline itself has no composite mission mixture or nonparametric size correction.

## 15. Complexity and fitted capacity in the current runner

The following counts are obtained from the parameter blocks described above. They exclude frozen preference controls, frozen parent models, and buffers. They count allocated trainable entries, not identifiable statistical degrees of freedom. In particular, dense pairwise AR allocates unused triangle entries.

| Baseline | ERIM: 28 products, 3,848 household IDs | Dunnhumby: 64 products, 2,495 household IDs | Main calculation beyond independent item scores |
|---|---:|---:|---|
| `popularity` | 28 | 64 | Factorized normalization |
| `no_price` | 28 | 64 | Supplied preferences |
| `independent_parent` | 56 | 128 | Learned own-price scores |
| `size_aware_esp` | 3,877 | 2,560 | Size softmax and catalogue ESP; fixed choice terms cached |
| `independent_categories_esp` | 28 | 64 | Per-category ESP and count/gate calculations |
| `mission_esp` | 7,752 | 5,118 | Two category-law components and household weights |
| `gaussian_factor_mixture` | 46,188 | 29,952 | Twelve fixed component likelihoods and household weights |
| `low_rank_bilinear_ar` | 280 | 640 | Factor-derived, currently dense prefix calculations |
| `pairwise_ar` | 812 | 4,160 | Dense preceding-product matrix |
| `neural_nade` | 1,848 | 4,224 | Width-16 hidden prefix and rank-8 price bridge |
| `structured_energy` | 92 | 211 | Count/graph partition dynamic program |

These numbers explain why names alone are misleading. The real Gaussian mixture fits more household weights than neural NADE fits parameters. Conversely, structured energy has relatively few learned coefficients but a more involved partition calculation. Caching can also make a parameter-heavy model's repeated fitting steps cheap.

Synthetic runs differ: there are only 17 household IDs, more household blocks are trainable, several baselines receive a separately fitted frozen independent parent, Gaussian loadings are learned, and low-rank AR uses rank 6. Results compare those implemented variants; they do not establish the best possible performance of each general model family.

### Baselines versus proposed-model ablations

`uncalibrated_mission_law` is an additional synthetic ablation of our proposed `CalibratedMissionLaw`. It uses the same two-mission, category-count, race-PL law with the hierarchical pooling penalties disabled. Ordinary ridge remains. It is not `mission_esp`, which has different category/count and subset-choice mechanics.

There are 11 baseline families overall. Real runs include all 11 plus the real proposed composite. Synthetic runs include 10 baselines, the proposed calibrated mission law, and its uncalibrated ablation. Each runner therefore has 12 entries, with different composition.

## 16. How to use the ladder scientifically

Each comparison asks a different question:

| Comparison | Question |
|---|---|
| Popularity versus preference-only | Are differences between shoppers/context important? |
| Preference-only versus independent price model | Does observed price information improve prediction? |
| Independent versus explicit-size/category models | Is basket-count structure missing? |
| Category model versus mission mixture | Does a shared hidden trip state explain co-purchasing? |
| Compact versus dense prefix effects | Are shared factors sufficient for product relationships? |
| Linear AR versus neural NADE | Do nonlinear conditional patterns improve fit or behavior? |
| AR versus structured energy | Does a joint count/interaction law improve the relevant outcomes? |

Except for carefully matched ablations, these comparisons can change multiple ingredients. The implemented variants also freeze different blocks. A difference in score should not automatically be attributed to one mechanism, and parameter counts do not replace a measured compute comparison.

A complete assessment uses:

1. **Prediction:** held-out basket likelihood, calibration, and recommendation recall.
2. **Generation:** product frequencies, basket-size distribution, category pairs, multiple purchases, and household variation.
3. **Synthetic what-if behavior:** own-product, same-category, and cross-category responses against the known generator.
4. **Synthetic decisions:** achieved contribution margin and regret under identical promotion choices and explicitly hypothetical costs/funding.
5. **Practical cost:** complete fitting/dependency time, memory, generation speed, and scenario cost at matched scales.

Co-purchasing can come from shared missions, counts, or nonempty conditioning. A pair coefficient, a latent factor, or a neural connection alone is not evidence of causal complementarity. Evaluate price interventions through the full fitted probability law.

### Running the models

See the [README](../README.md) for installation and data verification. To fit all applicable baselines and the proposed models:

```bash
python benchmarks/run_all_benchmarks.py --stage all --mode fresh --results-dir results/comparison_001
```

For a focused real-data comparison:

```bash
python benchmarks/run_all_benchmarks.py --stage erim --models independent_parent,size_aware_esp,neural_nade --mode fresh --results-dir results/focused_001
```

For synthetic pairwise and neural baselines:

```bash
python benchmarks/run_all_benchmarks.py --stage synthetic --world missions_b --models pairwise_binary_ar,neural_nade --mode fresh --results-dir results/focused_synthetic
```

Dependencies run first. Resume requires matching data, code, configuration, fixed state, fitting policy, and checkpoint checksums, then recertifies saved weights and recomputes evaluation. Convergence checks apply to active parameters; they do not prove a global optimum or identify true price effects.

The [historical tables](../results/historical) precede the corrected fresh-fit policy and include unconverged baselines and incomplete historical timing. Their ranking is provisional. Use outputs from the corrected pipeline for a new controlled comparison.
