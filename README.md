## Credit Scoring Business Understanding

Credit scoring is a critical function in financial institutions, as it directly impacts lending decisions, risk management, and regulatory compliance. This section explains the rationale behind our approach to credit risk modeling, taking into account the Basel II Accord, the need for proxy variables, and model selection trade-offs.

---

### Basel II Accord and Model Interpretability

The **Basel II Capital Accord** is a global regulatory framework that sets standards for measuring and managing credit risk in banks. Its primary objectives are to ensure banks maintain adequate capital to cover risks and to promote transparency and consistency in risk assessment.

**Implications for our model:**

- **Transparency:** The model must provide clear insights into how predictions are made. This ensures auditors, regulators, and internal stakeholders can understand and trust the risk scores.
- **Documentation:** Every step of data preprocessing, feature engineering, and modeling must be well-documented. This includes assumptions, transformations, and rationale for chosen methods.
- **Auditability:** Decisions based on the model must be explainable. For instance, if a customer is labeled as high-risk, the bank should be able to justify this using observable features.

In essence, regulatory compliance under Basel II mandates that our model is **both predictive and interpretable**, balancing statistical rigor with business accountability.

---

### Necessity of a Proxy Variable

Our dataset does not include a direct label for loan default. This is common in alternative data scenarios, where historical credit outcomes may be unavailable. To address this, we create a **proxy variable** to approximate credit risk.

**How we define the proxy:**

- Customers are segmented based on **transactional behavior** using RFM (Recency, Frequency, Monetary) analysis.
- Low engagement (e.g., infrequent transactions, low transaction amounts) is assumed to indicate higher credit risk.
- Clustering techniques like **K-Means** are used to identify groups of high-risk and low-risk customers.

**Business Risks of Using a Proxy:**

- **Misclassification:** Some customers labeled as high-risk may actually be low-risk, and vice versa. This can result in lost revenue or unnecessary denial of credit.
- **Bias:** If the proxy does not fully capture true default behavior, the model may inherit systematic biases.
- **Operational Risk:** Decisions based on an imperfect proxy must be carefully monitored to avoid adverse impacts on customer relationships and regulatory compliance.

Therefore, we treat the proxy variable as a **starting point** for credit scoring and plan to continuously validate and refine it.

---

### Trade-offs: Simple vs. Complex Models

In regulated financial contexts, the choice of model involves balancing **interpretability** and **predictive performance**:

- **Simple Models (e.g., Logistic Regression with Weight of Evidence):**

  - **Pros:** Easy to interpret, fully auditable, regulatory-friendly.
  - **Cons:** May struggle to capture complex non-linear relationships in the data.
  - **Use case:** Suitable for initial deployment and regulatory approval.

- **Complex Models (e.g., Gradient Boosting, Random Forests):**
  - **Pros:** High predictive accuracy, capable of modeling intricate patterns in large datasets.
  - **Cons:** Difficult to interpret, requires extensive documentation, potential regulatory scrutiny.
  - **Use case:** Can be used as an advanced scoring layer or for internal decision support.

**Strategy for this project:** We will experiment with both types of models. Initially, we prioritize interpretability to ensure compliance. Later, we explore high-performance models to improve predictive power while implementing methods to explain their outputs.

---

### Summary

This section demonstrates our understanding of **credit risk fundamentals** and the **regulatory and business context**. By combining regulatory knowledge, careful proxy creation, and a balanced modeling strategy, we aim to build a **robust, compliant, and actionable credit scoring system** for Bati Bank.
