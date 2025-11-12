# Mathematical Formulation of Tumor Growth Simulation

The tumor growth process is modeled as:
$$
Y_t = Y_{t-1} \times \left[ 1 + \rho \cdot \ln\left( \frac{K}{Y_{t-1}} \right) - \beta_c \cdot C_{t-1} - \left( \alpha \cdot d_{t-1} + \beta \cdot d_{t-1}^2 \right) + \epsilon_t \right]
$$

where:
- \(Y_t\): Tumor volume (outcome) at time \(t\).
- \(\rho\): Intrinsic growth rate factor.
- \(K\): Carrying capacity; the tumor volume corresponding to the death threshold.
- \(\beta_c\): Coefficient quantifying the effect of the chemotherapy dose.
- \(C_{t-1}\): Chemotherapy dosage applied at time \(t-1\). It is updated as:
  $$
  C_t = C_{t-1}\cdot \exp\left(-\frac{\ln2}{\text{drug\_half\_life}}\right) + (\text{new chemo dose})
  $$
- \(\alpha\): Linear effect coefficient for radiotherapy.
- \(\beta\): Quadratic effect coefficient for radiotherapy.
- \(d_{t-1}\): Radiotherapy dosage applied at time \(t-1\).
- \(\epsilon_t\): A noise term that introduces random variability.

Treatment assignments are generated using a sigmoid function. Define the cancer metric \(m_t\) as the average tumor diameter computed from the observations in the window from \(t-\text{window\_size}-\text{lag}\) to \(t-\text{lag}\). The probabilities for treatment assignment are then given by:

For chemotherapy:
$$
P(\text{chemo at } t) = \frac{1}{1+\exp\left(-\beta_{\text{chemo}} \cdot \left(m_t - \theta_{\text{chemo}}\right)\right)}
$$

For radiotherapy:
$$
P(\text{radio at } t) = \frac{1}{1+\exp\left(-\beta_{\text{radio}} \cdot \left(m_t - \theta_{\text{radio}}\right)\right)}
$$

where:
- \(\beta_{\text{chemo}}\) and \(\beta_{\text{radio}}\) control the sensitivity of the treatment assignment.
- \(\theta_{\text{chemo}}\) and \(\theta_{\text{radio}}\) are the sigmoid intercepts.
