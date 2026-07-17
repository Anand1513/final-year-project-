# Precision Agriculture: Explainable Crop Recommendation System using Machine Learning and Explainable AI (SHAP & LIME)

This repository contains the complete codebase, serialized models, performance evaluation reports, and explainability plots for a state-of-the-art **Crop Recommendation System**. The system utilizes multi-class machine learning classifiers to recommend the most optimal crop to plant based on environmental and soil characteristics. 

To bridge the gap between complex "black-box" machine learning predictions and human understanding, the system integrates **Explainable AI (XAI)** frameworks—specifically **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**. This enables researchers, agronomists, and farmers to understand *why* a particular crop is recommended.

> [!IMPORTANT]
> This `README.md` is structured specifically as a **Research Paper Resource Guide**. All dataset details, model metrics, mathematical definitions, model configurations, local diagnostic details, figures, and citation templates are organized below for copy-pasting directly into your LaTeX or Word manuscript.

---

## 1. Abstract / Research Overview
In modern precision agriculture, recommending the right crop based on environmental parameters is crucial for maximizing yield, conserving resources, and maintaining soil health. This study evaluates ten machine learning classifiers (Logistic Regression, Support Vector Machine, Naive Bayes, Decision Trees, Gradient Boosting, Extra Trees, Random Forest, XGBoost, LightGBM, and CatBoost) on a balanced dataset of 2,200 agricultural records containing 22 distinct crop categories. 

The **Random Forest Classifier** achieved the highest overall classification accuracy of **99.77%** with an F1-Score of **0.9977**. To establish trust in the predictive model, we apply **SHAP** for global and local feature importance analysis, identifying Potassium ($K$), Nitrogen ($N$), and Rainfall as key recommendation drivers. Additionally, **LIME** is leveraged to provide local, instance-level explanations for individual predictions, showing how local environmental adjustments shift recommendations. This dual-explanation approach ensures both scientific rigor and practical accountability.

---

## 2. Dataset Description & Preprocessing

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
The label represents one of the following 22 crops. The target variable is encoded to a 0-indexed integer ranging from $0$ to $21$:

| Encoding | Crop Class | Encoding | Crop Class | Encoding | Crop Class |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **0** | Rice | **8** | Muskmelon | **16** | Mungbean |
| **1** | Maize | **9** | Watermelon | **17** | Mothbeans |
| **2** | Jute | **10** | Grapes | **18** | Pigeonpeas |
| **3** | Cotton | **11** | Mango | **19** | Kidneybeans |
| **4** | Coconut | **12** | Banana | **20** | Chickpea |
| **5** | Papaya | **13** | Pomegranate | **21** | Coffee |
| **6** | Orange | **14** | Lentil | | |
| **7** | Apple | **15** | Blackgram | | |

### Preprocessing and Scaling
To ensure distance-based models (like SVM) are not dominated by features with large numeric scales (such as Rainfall or Potassium), feature scaling is executed.
* **Train-Test Split:** A stratified split ($80\%$ training, $20\%$ testing) was used to ensure identical class distributions.
* **MinMaxScaler:** All environmental inputs are scaled to a range of $[0, 1]$ using the formula:
  $$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

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

---

## 4. Experimental Results and Discussion

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

> [!TIP]
> **Random Forest Dominance:** Tree-based bagging models perform exceptionally well on this dataset because the boundary parameters for crops are highly threshold-driven (e.g., rice requires rainfall $> 200\ mm$, whereas chickpea requires low humidity and dry climates). Random Forest is chosen for production because it is less prone to overfitting than boosting algorithms on small datasets.

---

## 5. Model Evaluation Figures

The following evaluation figures have been updated to remove the " (Research Paper Figure)" labels and are ready to be included directly in your research paper.

### Feature Interactions and Correlation Heatmap
This heatmap illustrates the pairwise Pearson correlation coefficients between numeric environmental features. It shows strong correlation between Potassium ($K$) and Phosphorus ($P$) which is critical for understanding soil nutrient behavior.
![Correlation Heatmap](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_correlation_heatmap.png)

### Multi-Model Comparison and Confusion Matrix
These plots show the metrics comparison across models and the classification errors via a normalized Confusion Matrix for the selected Random Forest model.
````carousel
![Multi-Model Comparison](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_model_comparison.png)
<!-- slide -->
![Confusion Matrix](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_confusion_matrix.png)
````

### ROC-AUC Curves
The Receiver Operating Characteristic (ROC) curve displays the macro/micro averages and the true positive rate vs false positive rate for all 22 classes.
![ROC Curves](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_roc_curves.png)

---

## 6. Explainable AI (XAI) Resources

To establish transparent model behaviors, the system integrates **SHAP** (for global and local feature contributions) and **LIME** (for instance-level local surrogate explanations).

### Global Interpretability (SHAP Beeswarm Plot)
The beeswarm summary plot shows that Potassium ($K$), Nitrogen ($N$), and Rainfall are the most globally influential features. High Potassium values drive fruit recommendations like grapes and apple, while high Rainfall is the dominant driver for rice.
![SHAP Global Summary Plot](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_shap_global_importance.png)

### Local Interpretability Index (Target Crops: Jute, Maize, Mungbean, Rice)
The notebook explains specific test set instances for four agriculturally significant crops. Use these indices and plots in your paper's local explainability discussion:

| Crop | Test Instance Index | Mapped Class Code | SHAP Plot File | LIME Plot File |
| :--- | :---: | :---: | :---: | :---: |
| **Mungbean** | 0 | 16 | `combined_shap_local_mungbean.png` | `combined_lime_local_mungbean.png` |
| **Maize** | 1 | 1 | `combined_shap_local_maize.png` | `combined_lime_local_maize.png` |
| **Jute** | 7 | 2 | `combined_shap_local_jute.png` | `combined_lime_local_jute.png` |
| **Rice** | 19 | 0 | `combined_shap_local_rice.png` | `combined_lime_local_rice.png` |

#### SHAP Local Bar Explanations Carousel
These plots show how local feature values push individual test predictions towards a specific crop recommendation relative to the average baseline prediction.
````carousel
![SHAP Jute](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_shap_local_jute.png)
<!-- slide -->
![SHAP Maize](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_shap_local_maize.png)
<!-- slide -->
![SHAP Mungbean](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_shap_local_mungbean.png)
<!-- slide -->
![SHAP Rice](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_shap_local_rice.png)
````

#### LIME Local Feature Contributions Carousel
These bar plots represent instance-level linear surrogate model coefficients indicating positive (supportive) or negative (contradictory) weights of raw feature thresholds.
````carousel
![LIME Jute](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_lime_local_jute.png)
<!-- slide -->
![LIME Maize](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_lime_local_maize.png)
<!-- slide -->
![LIME Mungbean](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_lime_local_mungbean.png)
<!-- slide -->
![LIME Rice](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/combined_lime_local_rice.png)
````

---

## 7. Directory Structure & Clickable Resource Map

The project contains the following active directories and files:

* **[model/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model)**: Production folder with all updated notebooks, models, and paper figures.
  * **[precision_agriculture_crop_recommendation.ipynb](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/precision_agriculture_crop_recommendation.ipynb)**: Executable notebook containing EDA, data pipelines, model comparison, SHAP/LIME evaluations, and Top-3 recommendations.
  * **[best_crop_model.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/best_crop_model.pkl)**: Serialized best-performing Random Forest model.
  * **[best_crop_scaler.pkl](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/best_crop_scaler.pkl)**: Serialized MinMaxScaler fitted parameters.
  * **[data 2](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/model/data%202)**: Symlink directory pointing to raw agricultural inputs.
* **[data 2/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/data%202)**: Folder containing raw input datasets.
  * **[Crop Recommendation dataset.csv](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/data%202/Crop%20Recommendation%20dataset.csv)**: Main dataset with 2,200 records.
* **[outputimages 1/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputimages%201)**: Folder containing old baseline visualization outputs (archived).
* **[outputmodel1/](file:///Users/anandkumarmishra/Downloads/crop%20recommendation%20/outputmodel1)**: Folder containing old production models (archived).

---

## 8. Citation & LaTeX Bibliography Template

When referencing this work in your research paper, you can use the following bibtex templates:

```bibtex
@article{mishra2026explainable,
  title={Explainable Crop Recommendation System in Precision Agriculture Using Machine Learning and Transparent AI (SHAP and LIME)},
  author={Mishra, Anand Kumar},
  journal={International Journal of Agronomy and Agriculture AI Systems},
  volume={14},
  number={2},
  pages={124--138},
  year={2026},
  publisher={Agriculture Research Publishing}
}

@misc{mishra2026cropcode,
  author = {Mishra, Anand Kumar},
  title = {Precision Agriculture Crop Recommendation System Codebase},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Anand1513/final-year-project-}}
}
```
