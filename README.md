# Bank Customer Churn Prediction & Retention Analysis

## Overview
This project focuses on predicting bank customer churn using advanced ensemble machine learning techniques. Customer churn (attrition) is a critical challenge in the banking industry, where the cost of acquiring a new customer is significantly higher than retaining an existing one. The theoretical and practical goal of this project is to demonstrate how predictive analytics can be leveraged to analyze customer behavior, identify individuals at high risk of leaving the bank, and provide actionable insights for proactive retention strategies. 

This end-to-end project encompasses everything from raw data processing using Object-Oriented Programming (OOP) principles to deploying a final predictive model via an interactive web application.

---

## Goals
The primary goal of this project is to explore tabular data analysis and apply robust machine learning pipelines to solve a real-world business problem. This includes:

- **Comprehensive Data Analysis:** Investigating how sequential and static demographic data (like tenure, geography, and account balances) influence a customer's decision to close their account.
- **Advanced Predictive Modeling:** Applying and comparing powerful ensemble learning models, specifically Random Forest and Extreme Gradient Boosting (XGBoost), to accurately classify potential churners.
- **Software Engineering Best Practices:** Implementing the entire data preprocessing and model training pipeline using custom Python classes (OOP) to ensure modularity, scalability, and reusability of the code.
- **Model Deployment:** Transitioning the machine learning model from a Jupyter Notebook environment into a fully functional, interactive web interface using Streamlit, allowing stakeholders to perform real-time risk assessments.

---

## Data
This project utilizes a comprehensive bank customer dataset (`data_C.csv`) encompassing various demographic and financial metrics. The dataset provides a holistic view of the customer's relationship with the bank. The features are categorized as follows:

**Identifiers (Removed during preprocessing):**
- `id` / `CustomerId`: Unique identification numbers.
- `Surname`: Customer's last name.

**Demographics:**
- `Geography`: The customer's country of residence (France, Spain, Germany).
- `Gender`: Male or Female.
- `Age`: The customer's age.

**Financial & Engagement Metrics:**
- `CreditScore`: A numerical expression based on a level analysis of a person's credit files.
- `Tenure`: Number of years the customer has been with the bank.
- `Balance`: The amount of money in the customer's account.
- `NumOfProducts`: The number of bank products the customer utilizes (e.g., savings account, credit card, loan).
- `HasCrCard`: Binary flag indicating if the customer possesses a credit card (1 = Yes, 0 = No).
- `IsActiveMember`: Binary flag indicating if the customer is an active member (1 = Yes, 0 = No).
- `EstimatedSalary`: The customer's estimated annual salary.

**Target Variable:**
- `churn`: Binary variable indicating whether the customer left the bank (1 = Churned) or stayed (0 = Retained).

---

## Methodology
The project follows a rigorous, structured data science pipeline implemented via custom Python classes (`predata` and `Handling`):

### Data Preprocessing
Robust preprocessing was applied to ensure high-quality inputs for the machine learning models, reducing bias and preventing data leakage:
- **Feature Selection:** Dropping irrelevant features (`id`, `CustomerId`, `Surname`) that hold no predictive power and could introduce noise.
- **Categorical Encoding:** - Applying `LabelEncoder` to transform nominal features like `Geography` and `Gender` into machine-readable numerical formats.
  - Saving the fitted encoders as serialized files (e.g., `gender_encode.pkl`) to ensure consistency during the deployment phase.
- **Data Splitting:** Partitioning the dataset into training and testing sets to rigorously evaluate the models on unseen data.

---

## Modelling

### 1. Random Forest Classifier
The first approach utilizes a **Random Forest Classifier**, an ensemble learning method constructed using Scikit-Learn. 

The implementation includes:
- **Bootstrap Aggregating (Bagging):** Building multiple decision trees on random subsets of the data to significantly reduce the model's variance.
- **Feature Randomness:** Selecting a random subset of features at each split to ensure tree diversity and mitigate overfitting.
- **Non-linear Relationship Handling:** Capturing complex, non-linear interactions between financial indicators (like `Age` and `Balance`) without requiring extensive mathematical transformations.

This model serves as a strong, interpretable baseline for tabular classification.

### 2. XGBoost Classifier (Selected for Production)
The second approach implements **Extreme Gradient Boosting (XGBoost)**, recognized as an industry-standard algorithm for tabular data.

The implementation includes:
- **Gradient Boosting Framework:** Training decision trees sequentially, where each new tree attempts to correct the residual errors made by the previous ones.
- **Built-in Regularization:** Utilizing L1 and L2 regularization penalties to aggressively combat overfitting, resulting in a highly generalized model.
- **Optimal Performance:** Delivering superior execution speed and classification accuracy compared to standard bagging methods.

Due to its exceptional performance metrics on the test set, the XGBoost model was serialized (`xgb_classifier.pkl`) and selected as the core engine for the production environment.

---

## Model Evaluation & Deployment

### Evaluation
Both models were thoroughly evaluated using standard classification metrics to gauge their effectiveness in identifying the minority class (churners):
- **Classification Reports:** Analyzing Precision, Recall, and F1-Scores.
- **Focus on Recall:** Particular attention was paid to recall for the churn class, as failing to identify a churner (False Negative) is typically more costly for a bank than falsely identifying a loyal customer as a churn risk (False Positive).

### Web Deployment via Streamlit
To bridge the gap between theoretical data science and practical business application, the optimized XGBoost model was deployed using a **Streamlit** web application (`No_3.py`).
- **Interactive User Interface (UI):** Users can input specific customer data through sliders and dropdown menus (e.g., adjusting `Age`, selecting `Geography`).
- **Real-Time Inference:** Upon clicking "Make Prediction", the app preprocesses the inputs using the saved label encoders and passes the array to the XGBoost model.
- **Actionable Output:** The system instantly returns a clear business verdict: *"Customer is predicted to stay"* or *"Customer is predicted to churn"*.

---

## Future Work
This project establishes a solid foundation for predictive customer analytics. Future enhancements can include:
- **Hyperparameter Tuning:** Implementing `GridSearchCV` or `RandomizedSearchCV` to systematically explore the optimal parameter space for the XGBoost model (e.g., `max_depth`, `learning_rate`, `n_estimators`).
- **Class Imbalance Handling:** Integrating Synthetic Minority Over-sampling Technique (SMOTE) to synthetically generate examples of churned customers, potentially improving the model's sensitivity.
- **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) values into the Streamlit dashboard to provide stakeholders with visual explanations of *why* a specific model decision was made.

## Conclusion
This project demonstrates the powerful intersection of machine learning and business strategy. By methodically processing historical banking data, building object-oriented pipelines, and deploying an XGBoost model, this project provides a tangible, interactive tool that financial institutions can use to anticipate customer churn, optimize retention campaigns, and ultimately preserve revenue.
