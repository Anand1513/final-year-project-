# Precision Agriculture: Explainable Crop Recommendation System using Machine Learning and Transparent AI (SHAP & LIME)

This repository contains the complete codebase, serialized models, performance evaluations, and explainability plots for a state-of-the-art **Crop Recommendation System**. The system utilizes multi-class machine learning classifiers to recommend the most optimal crop to plant based on environmental and soil characteristics. 

To bridge the gap between complex "black-box" machine learning predictions and human understanding, the system integrates **Explainable AI (XAI)** frameworks—specifically **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**. This enables researchers, agronomists, and farmers to understand *why* a particular crop is recommended.

---

## 1. Abstract
In modern precision agriculture, recommending the right crop based on environmental parameters is crucial for maximizing yield, conserving resources, and maintaining soil health. This study evaluates several machine learning classifiers (Logistic Regression, Support Vector Machine, Naive Bayes, Decision Trees, Gradient Boosting, Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost) on a balanced dataset of 2,200 agricultural records containing 22 distinct crop categories. 

The **Random Forest Classifier** achieved the highest overall classification accuracy of **99.77%** with an F1-Score of **0.9977**. To establish trust in the predictive model, we apply **SHAP** for global and local feature importance analysis, identifying Potassium ($K$), Nitrogen ($N$), and Rainfall as key recommendation drivers. Additionally, **LIME** is leveraged to provide local, instance-level explanations for individual predictions, showing how local environmental adjustments shift recommendations. This dual-explanation approach ensures both scientific rigor and practical accountability.

---

## 2. Dataset Description and Analysis
The dataset contains **2,200 observations** of soil and weather metrics, representing a perfectly balanced distribution across **22 different crops** (100 samples per crop). 

### Environmental Features (Inputs)
Each observation includes 7 features:
1. **N (Nitrogen):** Ratio of Nitrogen content in soil ($mg/kg$).
2. **P (Phosphorus):** Ratio of Phosphorus content in soil ($mg/kg$).
3. **K (Potassium):** Ratio of Potassium content in soil ($mg/kg$).
4. **Temperature:** Ambient air temperature in Celsius ($^\circ\text{C}$).
5. **Humidity:** Relative humidity in percentage ($\%$).
6. **pH:** Soil pH value indicating acidity/alkalinity ($0$ to $14$).
7. **Rainfall:** Average annual or cycle rainfall in millimeters ($mm$).

### Target Variable (Output)
The label represents one of the following 22 crops:
* **Grains/Cereals:** Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil.
* **Fruits:** Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut.
* **Industrial/Other:** Jute, Cotton, Coffee.

### Exploratory Data Analysis (EDA) Highlights
* **Missing & Duplicate Values:** The dataset contains $0$ null values and $0$ duplicate records, ensuring clean inputs.
* **Data Balance:** Standard value counts reveal exactly 100 samples per class, preventing any majority-class bias.
* **Dataset Characteristics:**
  * Mean Nitrogen ($N$): $50.55\ mg/kg$ (Standard Dev: $36.92$)
  * Mean Phosphorus ($P$): $53.36\ mg/kg$ (Standard Dev: $32.99$)
  * Mean Potassium ($K$): $48.15\ mg/kg$ (Standard Dev: $50.65$)
  * Mean Temperature: $25.62^\circ\text{C}$ (Range: $8.82^\circ\text{C}$ to $43.68^\circ\text{C}$)
  * Mean Humidity: $71.48\%$ (Range: $14.26\%$ to $99.98\%$)
  * Mean pH: $6.47$ (Range: $3.50$ to $9.94$)
  * Mean Rainfall: $103.46\ mm$ (Range: $20.21\ mm$ to $298.56\ mm$)

---

## 3. System Architecture & Methodology

The workflow of the proposed system is outlined in the diagram below:

```mermaid
graph TD
    A[Agricultural Dataset] --> B[Data Preprocessing & Encoding]
    B --> C[Feature Scaling: MinMaxScaler]
    C --> D[Train-Test Split: 80/20]
    D --> E[Multi-Classifier Exploration & Baseline Models]
    D --> F[Advanced Ensemble & Boosting Models]
    E --> G[Model Evaluation & Metrics Compilation]
    F --> G
    G --> H[Selection of Best Model: Random Forest]
    H --> I[Serialize Model & Scaler (.pkl)]
    H --> J[Explainable AI Framework]
    J --> K[SHAP Global & Local Interpretability]
    J --> L[LIME Instance-Level Surrogate Model]
    H --> M[Predictive System: Top-3 Crop Recommendation]
```

### Preprocessing and Scaling
To ensure distance-based models (like SVM) are not dominated by features with large numeric scales (such as Rainfall or Potassium), feature scaling is executed.
* **Target Encoding:** The categorical target labels (crops) are mapped to integers ($1$ to $22$).
* **MinMaxScaler:** All environmental inputs are scaled to a range of $[0, 1]$ using the formula:
  $$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

---

## 4. Machine Learning Models
Ten distinct machine learning classifiers were evaluated to compare linear, distance-based, probabilistic, and tree-based ensemble paradigms.

1. **Logistic Regression:** A linear classifier that fits a multiclass softmax regression model.
2. **Support Vector Machine (SVM):** Uses radial basis function (RBF) kernels to find optimal decision boundaries in high-dimensional space.
3. **Naive Bayes:** A probabilistic classifier based on Bayes' Theorem assuming feature independence.
4. **Gradient Boosting:** An additive boosting ensemble that builds sequential weak decision trees.
5. **Decision Tree (Extra Trees):** Randomizes split selections to reduce variance compared to standard decision trees.
6. **Random Forest:** An ensemble bagging classifier that aggregates predictions from multiple decision trees.
7. **XGBoost (Extreme Gradient Boosting):** An optimized gradient boosting library implementing parallel tree boosting and regularization.
8. **LightGBM:** A fast gradient boosting framework using histogram-based algorithms and leaf-wise growth.
9. **CatBoost:** An advanced gradient boosting algorithm designed to handle categorical and numerical features efficiently with minimal tuning.

---

## 5. Experimental Results and Discussion

Model training was performed using an $80\%$ training split ($1760$ samples) and a $20\%$ testing split ($440$ samples). 

### Performance Comparison Matrix

| Model Class | Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear** | Logistic Regression | 0.9182 | 0.9200 | 0.9182 | 0.9180 |
| **Kernel/Distance** | Support Vector Machine (SVM) | 0.9682 | 0.9730 | 0.9682 | 0.9682 |
| **Probabilistic** | Naive Bayes | 0.9955 | 0.9958 | 0.9955 | 0.9954 |
| **Standard Ensemble** | Gradient Boosting Classifier | 0.9818 | 0.9843 | 0.9818 | 0.9819 |
| **Standard Ensemble** | Extra Trees Classifier | 0.8932 | 0.8950 | 0.8932 | 0.8925 |
| **Advanced Boosting** | XGBoost Classifier | 0.9886 | 0.9889 | 0.9886 | 0.9886 |
| **Advanced Boosting** | LightGBM Classifier | 0.9886 | 0.9890 | 0.9886 | 0.9887 |
| **Advanced Boosting** | CatBoost Classifier | 0.9932 | 0.9935 | 0.9932 | 0.9932 |
| **Bagging Ensemble** | **Random Forest Classifier (Tuned)** | **0.9977** | **0.9979** | **0.9977** | **0.9977** |

### Analytical Insights
* **The Random Forest Dominance:** Random Forest achieved near-perfect classification accuracy (**99.77%**). Tree-based bagging models perform exceptionally well on this dataset because the boundary parameters for crops are highly threshold-driven (e.g., rice requires rainfall $> 200\ mm$, whereas chickpea requires low humidity and dry climates).
* **Boosting Efficiency:** CatBoost, XGBoost, and LightGBM performed comparably ($> 98.8\%$), showing that ensemble decision paths accurately capture multi-dimensional environmental thresholds.
* **Extra Trees Variance:** The Extra Trees model suffered from high randomness in splitting, yielding lower accuracy ($89.32\%$) relative to Random Forest ($99.77\%$).

---

## 6. Explainable AI (XAI)
To make predictions trustworthy, the system integrates two key diagnostic methods:

### SHAP (SHapley Additive exPlanations)
Based on cooperative game theory, SHAP allocates an importance value to each environmental feature for a given prediction.
* **Global Importance:** Summary plots show that Potassium ($K$), Nitrogen ($N$), and Rainfall are the most globally influential features. For instance, high Potassium levels heavily trigger grape recommendations, while high rainfall dictates rice.
* **Local Interpretability:** For a specific test sample (e.g., predicted as *Mungbean* or *Muskmelon*), SHAP waterfall plots visualize exactly which features pushed the prediction away from the baseline average and why.
* **Visual Artifact:** See `outputimages 1/advanced_shap_global_importance.png` and `outputimages 1/shap_global_importance.png` to examine the global features beeswarm plot.

### LIME (Local Interpretable Model-agnostic Explanations)
While SHAP calculates contributions across all potential feature subsets, LIME approximates the model locally by generating perturbed data around a specific input sample. It fits a simple, interpretable linear model on these perturbations.
* **Use Case:** If a farmer receives a prediction for *Muskmelon*, LIME generates a local bar plot showing, for example, that `Humidity > 80.0%` had a positive weight of $+0.35$ while `pH = 6.5` contributed $+0.15$ to the prediction.
* **Visual Artifact:** See `outputimages 1/lime_local_explanation.png` and `outputimages 1/advanced_lime_local_explanation.png`.

---

## 7. Predictive System with Top-3 Recommendations
A key limitation of standard classifiers is that they output only a single recommendation. In real-world farming, offering a single crop can be risky (e.g., if market prices crash or seeds are unavailable). 

This system implements a **Top-3 Recommender Engine**:
1. It takes the scaled environmental vector.
2. Instead of calling `predict()`, it executes `predict_proba()`.
3. It sorts the probability list and returns the top 3 crops along with their associated confidence percentages.
4. *Example output:*
   * Recommendation 1: Rice (Confidence: 87.5%)
   * Recommendation 2: Jute (Confidence: 11.2%)
   * Recommendation 3: Maize (Confidence: 1.3%)

This approach gives farmers flexible options based on risk assessment.

---

## 8. Directory Structure and Serialized Files
The project directory is structured as follows:

* **[model1/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model1)**: Folder containing development notebooks and serialized files.
  * **[advanced_crop_recommendation.ipynb](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model1/advanced_crop_recommendation.ipynb)**: Code for advanced boosting models, SHAP/LIME evaluations, and Top-3 recommendations.
  * **[crop_recommendation.ipynb](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model1/crop_recommendation.ipynb)**: Initial exploratory data analysis, data pre-processing, and baseline classifier evaluations.
  * **[advanced_model.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model1/advanced_model.pkl)**: Serialized best-performing Random Forest model.
  * **[advanced_scaler.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model1/advanced_scaler.pkl)**: Serialized MinMaxScaler fitted parameters.
* **[outputimages 1/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201)**: Visualization artifacts for paper figures:
  * **[model_comparison.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/model_comparison.png)**: Bar plot comparing baseline models.
  * **[advanced_model_comparison.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/advanced_model_comparison.png)**: Bar plot comparing boosting and random forest models.
  * **[confusion_matrix.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/confusion_matrix.png)**: Multi-class confusion matrix displaying classification errors.
  * **[roc_curves.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/roc_curves.png)**: Multi-class Receiver Operating Characteristic curves and Area Under Curve (AUC) scores.
  * **[shap_global_importance.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/shap_global_importance.png)**: SHAP summary plots highlighting feature contributions.
  * **[lime_local_explanation.png](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201/lime_local_explanation.png)**: Single-instance LIME bar explanations.
* **[outputmodel1/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputmodel1)**: Production-ready models.
  * **[crop_recommendation_advanced_model.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputmodel1/crop_recommendation_advanced_model.pkl)**: Best RF model.
  * **[crop_recommendation_scaler.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputmodel1/crop_recommendation_scaler.pkl)**: Production scaling parameter model.

---

## 9. Citation & Usage in Research Papers
When incorporating this work into your research paper, the following structure is recommended:

1. **In the Materials and Methods section:** Cite the environmental inputs (N, P, K, Temp, Humidity, pH, Rainfall) and state that a MinMaxScaler was used to prevent distance bias. Mention the use of target encoding mapping 22 crops to unique indices.
2. **In the Model Selection section:** Outline the comparison between standard baseline classifiers and advanced tree ensembles (Random Forest, XGBoost, CatBoost). Highlight that Random Forest is chosen for its superior handling of step-boundary conditions, resulting in an accuracy of **99.77%**.
3. **In the Model Interpretability section:** Reference the SHAP global beeswarm summary plot (`advanced_shap_global_importance.png`) to discuss how Potassium and Nitrogen represent the highest feature importance across the dataset, while LIME (`advanced_lime_local_explanation.png`) is utilized for farm-level decision transparency.
4. **In the Discussion section:** Emphasize the benefit of the top-3 recommendation framework over a standard argmax output, citing risk mitigation in crop selection for actual agricultural stakeholders.
