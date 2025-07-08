# Credit Risk Probability Model for Alternative Data

## An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

---

## Task 1 - Understanding Credit Risk

### Credit Scoring Business Understanding.

In the world of finance, the fear of risk looms large, demanding sophisticated methods for its measurement and management. The Basel II Accord, a cornerstone of international banking regulation, fundamentally shapes how financial institutions approach this challenge, emphasizing the need for robust and transparent risk models. This necessity can be seen from the practical aspect of data availability and the inherent tension between model simplicity and predictive power.

#### 1. Basel II and the Imperative for Clarity:

The Basel II framework, particularly its Internal Ratings-Based (IRB) approach, empowers banks to develop their own models for estimating key risk parameters like the Probability of Default (PD). However, this autonomy comes with a significant regulatory burden. The Accord's "Pillar 2," the Supervisory Review Process, mandates that these internal models are not only accurate but also thoroughly documented and readily interpretable. As a result, regulators must be able to understand the model's logic, its assumptions, and its limitations to ensure it provides a sound basis for capital adequacy calculations. This requires a level of transparency that allows a knowledgeable third party to replicate the model's outcomes, fostering confidence in the institution's risk management practices. Consequently, a black box model, no matter how predictive, faces significant hurdles in gaining regulatory approval.

#### 2. The Challenge of "Default" and the Perils of Proxies:

When we consider the fundemtal challenge in credit risk modeling,it is the absence of clear-cut default label. While a loan is either in default or it is not, the data available to build predictive models may not always capture this binary outcome directly or in a timely manner. This necessitates the creation of a proxy variable, a stand-in for the true default event. For instance, a loan being a certain number of days past due can serve as a proxy for default.

However, relying on proxies introduces potential business risks. A poorly chosen proxy may not accurately reflect the true likelihood of default, leading to flawed predictions. For example, if a proxy is too lenient, the model may underestimate risk, leading to inadequate capital reserves and potential future losses. Conversely, an overly stringent proxy could lead to overly conservative lending practices, causing the institution to miss out on creditworthy customers. Furthermore, if the proxy variable is correlated with sensitive demographic or geographic information, it can introduce unintended bias into the model, leading to discriminatory lending decisions and significant reputational and legal risks.

#### 3.The Trade-Off: Simplicity vs. Complexity in a Regulated World:

In the regulated financial sphere, the choice of modeling technique presents a critical trade-off between interpretability and predictive performance.

- **Simple, Interpretable Models such as Logistic Regression with Weight of Evidence - WoE:**
  These models have long been favored in the financial industry for their transparency. Logistic Regression produces easily understandable coefficients that quantify the impact of each variable on the probability of default. The use of WoE transformations further enhances interpretability by creating a monotonic relationship between the input variables and the outcome. This clarity facilitates communication with regulators, auditors, and business stakeholders, making it easier to justify lending decisions and demonstrate compliance. But this models have drawback that they may not capture complex, non-linear relationships in the data, potentially sacrificing some predictive accuracy.
- **Complex, High-Performance Models (e.g., Gradient Boosting):**  
   Techniques like Gradient Boosting are renowned for their superior predictive power. They can uncover intricate patterns and interactions within the data that simpler models might miss, leading to more accurate risk assessments. However, this performance comes at the cost of interpretability. The inner workings of these "black box" models are often opaque, making it difficult to explain why a particular prediction was made. In a regulated context, this lack of transparency is a significant hurdle. While emerging techniques in "Explainable AI" (XAI) are beginning to lessen this gap by providing insights into the decision-making processes of complex models, still the regulatory landscape still generally favors the established transparency of simpler approaches.
  Finally, financial institutions must navigate a delicate balance. While the allure of higher accuracy from complex models is strong, the stringent regulatory requirements for transparency and interpretability under frameworks like Basel II often lead to a continued reliance on simpler, more explainable models. The ideal solution lies in finding the right equilibrium—a model that is both sufficiently predictive to manage risk effectively and transparent enough to satisfy regulatory scrutiny.

## Task -2 Exploratory Data Analysis (EDA)

After doing the exploratory data analysis the following key insights are observed

**Top Insights made from EDA**

- Dominance of Small Transactions: Most transactions are small (below 10K UGX), primarily in 'airtime' and 'financial_services', indicating a focus on mobile-related services.
- Fraudulent Transactions are Rare but High-Value: Fraudulent transactions (FraudResult = 1) are a small fraction but involve significantly higher amounts (e.g., 700K, 725K UGX), suggesting a pattern of high-value fraud in 'financial_services'.
- Strong Correlation Between Amount and Value: The near-perfect correlation (0.95) between Amount and Value indicates redundancy, suggesting Value could be dropped or used differently in modeling. - Skewed Distributions and Outliers: Both Amount and Value are right-skewed with significant outliers, which may require transformation (e.g., log-scaling) or special handling in modeling. - Temporal Patterns: Transaction volume shows periodic spikes, potentially tied to billing cycles or promotions, suggesting time-based features (e.g., hour, day) could enhance fraud detection.

## Task-5: Model Training

After training with provided dataset using the pipeline the following log result found using the three models.

Data shape: (95662, 36), Target shape: (95662, 1)
Numeric columns for training: ['Amount', 'Value', 'TransactionHour', ..., 'ProviderId_ProviderId_6']
Train shape: (76529, 27), Test shape: (19133, 27)
Training LogisticRegression
Best parameters for LogisticRegression: {'C': 1, 'penalty': 'l2'}
Metrics for LogisticRegression: {'accuracy': 0.99, 'precision': 0.99, ‘f1’:0.99, ‘recall’:0.99, ‘roc_auc’:0.99}
Training RandomForest
Best parameters for RandomForest: {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
Metrics for RandomForest: {'accuracy': 0.98, 'precision': 0.95, ‘f1’: 0.95, ‘recall’:0.98, roc_auc: 0.98}
Training GradientBoosting
Best parameters for GradientBoosting: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3}
Metrics for GradientBoosting: {'accuracy': 0.95, 'precision': 0.92, ‘f1’: 0.95, ‘recall’:0.95, roc_auc: 0.95}
Registered best model: Logistic Regression with F1 score: 0.99

## Project Structure

<pre>
Credit-Risk-Model-Automation/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
|   ├── README.md
|   ├── 10-fe_proxy.ipynb     # Proxy variable FE pipeline
|   ├── 10-fe.ipynb           # Feature Engineering pipeline
│   └── 10-eda.ipynb          # Exploratory, one-off analysis
├── scripts/
|   ├── __init__.py 
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for Data Processing (EDA)
|   ├── data_processing_FE.py     # Feature Engineering (FE)
|   ├── data_processing_FE_Proxy.py # Script Proxy FE 
|   ├── model_training.py           # Script Model Training 
│   └── api/
│       └── __init__.py #
├── tests/
|   ├── __init__.py
|   ├── test_model_training.py # unit tests
│   └── test_sample.py         # Unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
</pre>

## Getting Started

1. Clone the repository

- git clone http://github.com/tegbiye/Credit-Risk-Model-Automation.git
- cd Credit-Risk-Model-Automation

2. Create environment using venv
   python -m venv .venv

- Activate the environment

  .venv\Scripts\activate

  source .venv\bin\activate

3. Install Dependencies

pip install -r requirements.txt

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with proper attribution.
