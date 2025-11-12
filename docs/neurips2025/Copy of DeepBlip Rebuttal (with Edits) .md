# Neurips 2025 Deepblip Rebuttal

# Response to Reviewer 1 MoAN

We thank you for your detailed and constructive feedback, which helps us clarify our contributions and your concerns. We will incorporate all improvements labeled with “**Action**” into our revised paper.  

## W1: Novelty and technical contributions

We respectfully disagree with the assessment that our technical contributions are trivial. While our work builds on the theoretical foundation of Structural Nested Mean Models (SNMMs), we emphasize that SNMMs are purely a theoretical construct and **not an instantiable model**. Existing SNMM-based approaches like Lewis & Syrgkanis \[3\] only implement SNMMs in a **linear setting with manually solved moment equations**, and do **not** support personalization via complex patient histories nor scalable training.

In contrast, our work makes several non-trivial contributions that significantly advance this line of research:

* We introduce the first neural instantiation of SNMMs, enabling flexible function approximation for complex, high-dimensional histories (e.g., from medical time-series).  
* We design a two-stage sequential neural architecture (cf. Fig. 3), including tailored encoders and prediction heads to estimate nuisance functions and blip coefficients.  
* We reformulate the g-estimation problem as a moment-based loss minimization task that supports end-to-end training via backpropagation.

We believe these innovations align closely with NeurIPS’s mission to advance learning methods that are both principled and practical. Further, we have explicitly acknowledged prior work like \[3\] and positioned our method as a neural generalization of SNMMs with enhanced scalability and temporal stability.

Finally, we provide a direct comparison with Lewis & Syrgkanis (2021) in a controlled experiment, where our model significantly outperforms theirs across all metrics and settings:

We **compare below the performance of DeepBlip against the Dynamic Double/Debiased Machine Learning (DML) method from Lewis & Syrgkanis (2021)**. The experiment assesses the accuracy of each method in estimating the CATE over time, under conditions of increasing confounding and increasing prediction horizons. The two treatment sequences are $a \= (1, 1, .., 1)$ and $b \= (0, 0, .., 0)$. The CATE over time is measured at time $t+\\tau$, where $\\tau$ represents the prediction horizon. To accommodate the fixed requirement of dynamic DML, we fix the observable time $t$ to be 5\. 

The evaluation metric is the RMSE between the estimated CATE and the ground-truth CATE.

The Dynamic DML method (heterogeneous version) is implemented as originally designed, relying on sequential g-estimation. To enable comparison on the benchmark, we adapt the Dynamic DML to accept the history variable $H\_t$ as the conditioning variable, instead of the original static covariate. To do this, we concatenate all previous variables included in $H\_t$ into a single vector variable $V$. This variable $V$ serves as the new static covariate for dynamic DML. 

### **Task1**

This task evaluates model robustness to increasing levels of time-varying confounding using the **Tumor Growth** Dataset. The prediction horizon is fixed at $\\tau=2$. 

The results show that, while both methods are affected by stronger confounding, **DeepBlip consistently achieves a lower RMSE**. 

| Method\\Conf lv. | $\\gamma=2$ | $\\gamma=4$ | $\\gamma=6$ | $\\gamma=8$ | $\\gamma=10$ |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Dynamic DML | 0.53 | 0.57 | 0.62 | 0.66 | 0.77 |
| DeepBlip | **0.46** | **0.51** | **0.54** | **0.57** | **0.70** |

### **Task 2**

This task evaluates model stability over longer time horizons using the **MIMIC-III Semi-Synthetic** Dataset. The confounding strength is fixed to be 1\. In this dataset, the covariate has a **high dimension** of 25\.

The results demonstrate that **DeepBlip has a better performance in the presence of high-dimensional covariates, especially as the prediction horizon extends**. 

| Method\\horizon | $\\tau=2$ | $\\tau=4$ | $\\tau=6$ | $\\tau=8$ | $\\tau=10$ |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Dynamic DML | 0.45 | 0.64 | 0.77 | 0.92 | 1.16 |
| DeepBlip | **0.39** | **0.56** | **0.70** | **0.82** | **0.93** |

## 

## W2: On missing technical details

* W2.1: We apologize if the definitions were not sufficiently clear. The variables $\\tilde{Y}*{t,k}$ and $\\tilde{A}*\_{t,j,k}$ represent residuals defined explicitly in Eq.(6). **Action:** We will move the definition of them earlier to improve readability.  
* W2.2: We provide extensive empirical validation for this trick in our ablation studies. In Appendix C.1, we show that DeepBlip-WDO (Without Double Optimization) performs significantly worse than our full model. Furthermore, in Supplement C.2, we provide a visual analysis showing that the double optimization trick leads to significantly less biased estimations of the blip coefficients. **Action:** We will provide a summary of the results in the main paper.   
* W2.3: Our methods adjust for time-varying confounding naturally through the SNMM framework. Eq.3 strictly estimates the CATE with the blip coefficients under the correct parametrization. With Remark 1, our method can estimate the blip coefficients accurately. Hence, the CATE could be estimated without bias. **Action:** We will place greater focus on Eq 3 for estimating blip coefficients to clarify how adjustments for time-varying confounding are done in our method..

## W3: On experiments

Thank you for pointing out areas where we can improve the clarity. We will revise the paper to include more details about the experiments.

* W3.1: We clarify that the method from \[3\] is not suitable for the setting of CATE over time estimation with dynamic history variable $H\_t.$. Nevertheless, we add experiments with fixed number of time steps $t$ to satisfy \[3\]’s requirement and compare our methods against the \[3\] on tumor growth dataset (against increasing confounding) and on MIMIC dataset (against increasing \\tau and high-dimensional covariates). For details about experiments please refer to W1. **Action**: We will complement this experiment in the supplement.  
* W3.2: This is an interesting proposal. However, static models are not designed for sequential treatments. We can forcefully treat the entire treatment sequence as a single high-dimensional treatment in order to use the static models. But this is exactly what we do in the HA-TRM model, which underperforms other baselines.  
* W3.3: We originally designed the experiments such that the tumor growth experiment evaluates the capability of addressing time-varying confounding and the MIMIC-III evaluates the stability over time horizons. We realize that this asymmetric design tends to raise concerns. The design choice is only due to the limited computing resources and it is possible to test robustness against increasing confounding and increasing horizon on both datasets. **Action**: We are working on the experiments to ensure that we test both the robustness over increasing confounding and over increasing horizon on all the datasets. We will present the full results in the revised version.   
* W3.4: We apologize for this omission from the main text. Due to limited space, we decided to put them in the appendices. For MIMIC-III, the covariates are 25-dimensional, as mentioned in Appendix D.2. The treatment is 2D binary, and the outcome is a hypothetical simulated continuous variable. **Action:** We will move the details to the main paper. 

## Q1: Weakness

We kindly refer to our answers above, where we have address your questions. 

## Q2: Challenges when applying neural networks to SNMM

We thank the reviewer for this question. The primary challenges are:

* Accelerating the training of second stage. The g-estimation procedure for SNMMs is originally sequential. To estimate the blip coefficient at step k, one must know the true blip coefficients for all future steps j \> k, which is computationally prohibitive. We introduce the double optimization trick to enable the simultaneous training of all the blip coefficient predictors.  
* Encoding the long-term histories. At first, we tried to use LSTM to encode the patient histories. However, LSTM is not stable to train over longer time series. We resolve this challenge by replacing LSTM with a transformer that is more powerful.

# Response to reviewer KUGC

We thank you for the insightful feedback. We will address your questions and the points raised in the weaknesses section below. We will incorporate all improvements labeled with “**Action**” into our revised paper.

## W1: Linear Blip parametrization

You are correct that our current implementation uses a the linear parametrization t,k (ht)$\\psi\_{t,k}(h\_t)’ a\_{t+k}$. We chose this formulation mainly for **clarity**. In fact, our framework can be readily extended to capture nonlinear treatment interactions by incorporating a nonlinear feature map $\\phi(\\cdot)$.   
The blip function can be then parametrized as $\\gamma\_{t,k}(h\_t) \= \\psi\_{t,k}’\\phi(a\_{t+k})$. Here $\\phi$ can be high-dimensional and include nonlinear features such as $(a\_{t+k}^{1})^2$ or $a\_{t+k}^{1} \* a\_{t+k}^{2}$. **Action:** We will mention this in the main text and acknowledge the current implementation as a specific instance of a more general framework.

## W2: On the Double Optimization Trick

The main purpose of the double optimization trick is to overcome the sequential dependency in the g-estimation. However, we provide **strong empirical evidence** in the appendix demonstrating that the scheme is stable. In Appendix C.1 (Figures 6 & 7), our ablation study on **`DeepBlip-WDO`** (Without Double Optimization) shows that removing this trick leads to significantly higher estimation error. More directly, in Appendix C.2 (Figures 8 & 9), we visualize the **distribution of the prediction error** for each individual blip coefficient against the ground truth. These figures clearly show that our full DeepBlip model produces accurate estimates centered at zero. 

**Action:** We will move the ablation against DeepBlip-WDO to our main paper to strengthen the empirical evidence supporting the choice of the double optimization trick. 

## W3: Missing literature

Thank you for pointing out these related works, which will help to strengthen our literature review. We will provide a discussion about these works below:

* Paper 1 is an orthogonal research streamline that estimates HTEs or potential outcomes over **continuous** time. In contrast, our aim is orthogonal because we focus on *discrete* time.   
* Paper 2 introduces the COSTAR framework that first pre-trains a transformer-based encoder using a self-supervised loss and then fine-tunes the encoder to predict the factual outcome. The work lacks a causal adjustment mechanism, hoping that the pre-trained transformer can incidentally transfer to the setting of estimating counterfactual outcomes. Hence, the approach is a heuristic and does **not** properly adjust for time-varying confounding.  
* Paper 3 and paper 4 are based on the balanced representation, like the causal transformer. Paper 3 adopts an adversarial generative approach to learn the balanced representations, which reduces the confounding bias. Paper 4 introduces a de-correlation strategy that removes cross-covariance between current treatments and representation of past trajectory, based on a data-selective state-space model architecture called Mamba. However, balanced representation-based methods are considered **biased** because they are only a heuristic that lacks a formal theoretical guarantee for adjusting for confounding. The underlying reason is that these methods were originally designed to reduce finite-sample variance, **not to mitigate confounding bias**. In fact, the process of enforcing such balancing would even introduce more bias \[3\].

To add on the arguments why balanced representations are biased: This was noted in the works on balanced representation \[1\]\[2\]\[4\], where balanced representation methods were primarily designed for variance reduction and act as heuristics for adjusting confounding. Moreover, it has been shown that if the representations are not invertible, enforcing the balancing constraints will even lead to representation-induced confounding bias (RICB) \[3\]. Therefore, methods based on balancing representations are inherently biased, unlike causal frameworks, such as G-computation or SNMMs. 

**Action**: We will present a more comprehensive literature review that includes these works in the revised version.

\[1\] Bica I. et al. “Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations” (ICLR 2020).  
\[2\] Melynchuk V. et al. “Causal Transformer for Estimating Counterfactual Outcomes” (ICML 2022).  
\[3\] Melynchuk V. et al. “Bounds on representation-induced confounding bias for treatment effect estimation” (ICLR 2024\)  
\[4\] Johansson et al. “Learning representations for counterfactual inference” (ICML 2016\)

## Q1: Addressing potential non-linear history-treatment interactions

Our framework addresses non-linearity in two ways:

1. Non-linearity in history $h\_t$: The function $\\psi\_{t,k}(h\_t)$ is parametrized by a powerful neural network (transformer), which allows our DeepBlip to capture non-linear relationships within the patient history $H\_t$. When conditioned on $H\_t$, it is easier to use the linear interactions to approximate a non-linear relationship.   
2. Adding non-linearity to the treatment vector. As stated in our response to W1, our framework can be extended to replace the treatment vector $a\_{t+k}$ with a non-linear feature map $\\phi(a\_{t+k})$.

**Action:** We will add a dedicated paragraph to our main paper where we explain that. How our framework addresses non-linearities.  

## Q2: Theoretical justification for the double optimization trick

The double-optimization trick is a practical and necessary algorithmic solution to the inherently sequential g-estimation scheme. To estimate the blip coefficient at step $k$, one needs the true coefficients from all future steps $j \> k$. However this is too time-consuming, and hence impossible to implement with scale. A trivial approach would be to add all the steps’ loss $L\_{k}$ ($k=0,..,\\tau$) together into a total loss $L$ and then directly optimize L. However, this approach discards the sequential order of the blip coefficients and, as we have shown empirically, produces biased blip coefficient estimates.

Our trick resolves this by using a second, detached forward pass to generate stable “pseudo-targets” for the future blip effects.   
The full learning process is anchored at the final time step $k=\\tau$, where there are no future steps. As $\\psi\_{t, \\tau}$ becomes accurate, it provides a high-quality target for learning $\\psi\_{t, \\tau \- 1}$, and so on. This allows the correct pattern to propagate backward implicitly. In sum, with the double optimization trick, we manage to train the $\\tau \+ 1$ blip coefficient estimators in parallel **while preserving the inherent sequential order of the g-estimation**.

For empirical evidence supporting the effectiveness of double optimization trick, we kindly ask you to refer to W2. 

# Response to reviewer 1pGk

We sincerely thank you for this valuable feedback. Your question about the theoretical justification for Remarks 2 and 3 is insightful and prompted a final review of our framework.

Upon re-evaluating our proofs, we identified an issue in the derivation for the L1-loss case. To ensure the theoretical rigor of our method, we have updated our implementation to use a standard **L2-moment loss**, which is well-established in the literature and has formal guarantees. We updated all experiments in the submitted material to reflect this change, but were unable to edit the main text before the deadline. We apologize for this discrepancy.

The L2-moment loss for each step k is:

$$\\mathcal{L}\_{blip}^{k} \= \\frac{1}{N} \\sum\_{i=1}^{N} \\left( \\tilde{Y}\_{t,k}^i \- \\sum\_{j=k+1}^{\\tau} \\hat{\\psi}\_{t,j}^{2}(H\_t^i)'\\tilde{A}\_{t,j,k}^i \- \\hat{\\psi}\_{t,k}^{1}(H\_t^i)'\\tilde{A}\_{t,k,k}^i \\right) ^2 $$

This change allows us to directly inherit the established theoretical properties for our framework. To clarify, here are the theorems corresponding to our remarks:  

**Remark 2 (Neyman Orthogonality)** The L2-moment loss is Neyman-orthogonal with respect to the nuisance functions ($p\\\_{t,k}$ and $q\\\_{t,j,k}$). This means that the estimation of the causal blip coefficients is robust to first-order errors in the estimation of the nuisance functions, a property also known as double robustness.  

**Remark 3 (Convergence Guarantee)**: Under standard regularity assumptions, the blip coefficient estimators are consistent and have a finite-sample mean squared error (MSE) guarantee. Adapted from Theorem 10 in Lewis & Syrgkanis (2021), this guarantee is:   
$$\\max\\\_{t,k} \\mathbb{E}\[||\\hat{\\psi}\*{t,k} \- \\psi\*{t,k}||\*{2,2}^{2}\] \= O(r^{2}\\delta\*{n}^{2}), \\quad \\text{where} \\quad \\delta\\\_{n}^{2} \\propto \\frac{\\log\\log(n)}{n}  
$$  
This ensures that our estimator converges to the true blip coefficients at a near-parametric rate.

The formal proofs for these results under an L2-loss are provided in Lewis & Syrgkanis (2021) (Theorem 10\) and can be extended to our neural setting. We hope this clarification addresses your concerns in W1, Q1, and Q2.

**Action**: We will incorporate these formal statements together with the proofs into the main text in the final version. 

## W2: Overstating limitations of CRN/CT

Thank you for mentioning this nuanced point. We agree that CRN and CT can empirically mitigate confounding. **We did not intend to claim they are generally ineffective but to show a theoretical disadvantage**. As noted in the works on balanced representation \[1\]\[2\]\[4\], balanced representation methods were primarily designed for variance reduction and act as heuristics for adjusting confounding. Moreover, it has been shown that if the representations are not invertible, enforcing the balancing constraints will even lead to representation-induced confounding bias (RICB) \[3\]. Therefore, methods based on balancing representations are very likely to be inherently biased, unlike causal frameworks, such as G-computation or SNMMs. 

In contrast, *our method based on SNMMs provides a theoretically grounded asymptotic convergence under standard assumptions*.  

**Action**: We will revise the text to reflect this distinction between empirical effectiveness and theoretical guarantees.

\[1\] Bica I. et al. “Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations” (ICLR 2020).  
\[2\] Melynchuk V. et al. “Causal Transformer for Estimating Counterfactual Outcomes” (ICML 2022).  
\[3\] Melynchuk V. et al. “Bounds on representation-induced confounding bias for treatment effect estimation” (ICLR 2024\)  
\[4\] Johansson et al. “Learning representations for counterfactual inference” (ICML 2016\)

## Q3: Justification for why DeepBlip works well for long horizons

The key justification is that our **DeepBlip avoids the error accumulation** that undermines competing methods. The reason is the following:

* Instead of modeling the full, complex counterfactual trajectory over a long horizon (as in G-computation ), DeepBlip breaks the problem down. It estimates the CATE as a sum of individual, time-specific "blip effects".  
* Each blip function isolates the incremental causal effect of a single treatment at a single point in time, conditioned on the history up to that point.  
* Because the estimation of these blips does not depend on modeling future outcomes or covariates, **estimation errors do not propagate over time**. This makes the learning task fundamentally more stable, especially as the prediction horizon $\\tau$ increases, which is confirmed by our empirical results in Table 2\.

**Action:** We will expand our motivation of using DeepBlip for long horizons. 

# Response to reviewer zUoX

We thank the reviewer for their positive evaluation and valuable feedback. We will address your questions below.

## Typos

Thank you for pointing out this notational mistake. The nuisance function $q\_{t,j,k}$ is trained to predict the conditional mean of observed treatment $A\_{t+j}$, as specified in our training procedure. Equation (8) should therefore be $\\mathbb{E}\[A\_{t+j} \\mid H\_{t+k} \= h\_{t+k}\]$. The use of $Q\_{t,j}$ was originally meant for a more general representation. However, for clarity, we simply the setting but forgot to change the notation. 

**Action:** We will fix this in the next version.

## Q1: Is the linear parametrization necessary?

While the interaction between the blip coefficients and the treatment needs to be a vector product, the treatment vector $a\_{t+k}$ can be readily extended to a non-linear feature map $\\phi(a\_{t+k}, h\_t, \\ldots)$. This allows for rich nonlinear interactions.

## Q2: Could performance difference due to assumption mismatches? 

Thank you for your insightful question. The SNMM indeed has slightly stronger model assumptions compared to completely non-parametric methods like G-computation, aside from the standard identifiability assumptions (consistency, positivity, and sequential ignorability). In contrast to G-computation, the SNMM has various other practical advantages such as better interpretability (e.g., one can easily interpret the incremental effect of each treatment at each time step). 

 For the estimation to be completely unbiased, the data model needs to satisfy the linear blip parametrization as discussed in Q1. However, we do not require the observational dataset to strictly follow this rule. The tumor growth dataset used in our paper in fact does not satisfy the linear blip assumption. Our DeepBlip still outperforms other baselines. Therefore, the superior performance of DeepBlip does not stem from making stronger assumptions, but comes from its inherent robustness over stronger confounding or longer horizons.

**Action**: We will state the assumptions in a clearer way in the revised version.

# Response to reviewer 11ry

We thank the reviewer for your detailed feedback. We appreciate the opportunity to address your concerns, and we are confident to resolve these issues in the revised version.

## W1: Theoretical proofs

Upon re-evaluating our proofs, we identified an issue in the derivation for the L1-loss case. To ensure the theoretical rigor of our method, we have updated our implementation to use a standard **L2-moment loss**, which is well-established in the literature and has formal guarantees. We updated all experiments in the submitted material to reflect this change, but were unable to edit the main text before the deadline. We apologize for this discrepancy.

The L2-moment loss for each step k is:

$$\\mathcal{L}\_{blip}^{k} \= \\frac{1}{N} \\sum\_{i=1}^{N} \\left( \\tilde{Y}\_{t,k}^i \- \\sum\_{j=k+1}^{\\tau} \\hat{\\psi}\_{t,j}^{2}(H\_t^i)'\\tilde{A}\_{t,j,k}^i \- \\hat{\\psi}\_{t,k}^{1}(H\_t^i)'\\tilde{A}\_{t,k,k}^i \\right) ^2 $$  
This change allows us to directly inherit the established theoretical properties for our framework. To clarify, here are the theorems corresponding to our remarks: 

**Remark 2 (Neyman Orthogonality)** The L2-moment loss is Neyman-orthogonal with respect to the nuisance functions ($p\\\_{t,k}$ and $q\\\_{t,j,k}$). This means that the estimation of the causal blip coefficients is robust to first-order errors in the estimation of the nuisance functions, a property also known as double robustness.  

**Remark 3 (Convergence Guarantee)**: Under standard regularity assumptions, the blip coefficient estimators are consistent and have a finite-sample mean squared error (MSE) guarantee. Adapted from Theorem 10 in Lewis & Syrgkanis (2021), this guarantee is:   
$$\\max\\\_{t,k} \\mathbb{E}\[||\\hat{\\psi}\*{t,k} \- \\psi\*{t,k}||\*{2,2}^{2}\] \= O(r^{2}\\delta\*{n}^{2}), \\quad \\text{where} \\quad \\delta\\\_{n}^{2} \\propto \\frac{\\log\\log(n)}{n}  
$$  
This ensures that our estimator converges to the true blip coefficients at a near-parametric rate.

The formal proofs for these results under an L2-loss are provided in Lewis & Syrgkanis (2021) (theorem 10\) and can be extended to our neural setting. We hope this clarification addresses your concerns about the missing proofs. 

**Action**: We will incorporate these formal statements into the main text in the revised version. 

## W2: Structural and presentation problem

We are grateful for your detailed proofreading and agree that the manuscript needs a thorough revision. We will address every point:

1. **Repetitive writing & highlighting**: You are right. We will perform a comprehensive revision to improve the writing flow, remove redundant statements, and use a single, standard form of emphasis to improve readability.  
2. **Presentation order & typos**: We will fix the issue of variables being used before their definition and correct all typos you have listed, in addition to conducting a full proofread of the entire paper. ($\\gamma\_{conf}$ is the parameter representing the strength of time-varying confounding.)  
3. **NeurIPS Style Guide**: Thank you for pointing this out. We will carefully reformat all tables, captions, references, and headings to strictly comply with the NeurIPS style guide. The reference list will be unified and completed.

Thank you\!

## 