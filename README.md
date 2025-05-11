## 1. Project Background and Dataset Overview

### Project Background

* We discuss how to leverage data intervention techniques and methods from the literature to improve model fairness across different groups (e.g., gender).
* The task objective is to predict whether income exceeds 50K based on demographic and economic features, while focusing on performance differences between genders.

### Dataset Overview

* The dataset includes the following features: `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, and the target variable `income`.
* By inspecting data types and previewing the first few rows, we can gain an initial understanding of the data’s structure and quality. For example:

  * Data types reveal the distribution of numerical and categorical variables.
  * The first few rows help us understand the basic sample data.

---

## 2. Data Preprocessing and Feature Engineering

### Target Variable Transformation

* Binarize the `income` column: map `>50K` to 1 and all others to 0 for easier model construction.

### Missing Value and Distribution Checks

* Check for missing values in each column and review unique values in `race` and `sex`, with special attention to gender distribution, to support later fairness analysis.

### Feature Engineering and Pipeline Construction

* Standardize numerical features (e.g., `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) using `StandardScaler`.
* Encode categorical features (e.g., `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) using `OneHotEncoder`.
* Combine these preprocessing steps with `ColumnTransformer`, then chain the preprocessing and a `LogisticRegression` model using `Pipeline` to ensure consistent data handling and streamlined model training.

---

## 3. Baseline Model Development and Evaluation

### Data Splitting and Model Training

* Split the dataset into features (`X_train`, `X_test`) and target (`y_train`, `y_test`).
* Train a baseline `LogisticRegression` model using the constructed pipeline and make predictions on the test set.

### Model Evaluation

* Evaluate the model using accuracy and the confusion matrix.
* Focus on the fairness metric—TPR (recall)—by calculating TPR separately for males and females.
* Results:

  * **Baseline Accuracy**: 0.8530
  * **Confusion Matrix**:

    ```
    [[11586   849]
     [ 1544  2302]]
    ```
  * **TPR**:

    * Male TPR: 0.6118
    * Female TPR: 0.5254

---

## 4. Controlled Sampling Intervention

### Motivation

* The original training set has far more male samples (21,790) than female samples (10,771), which may bias the model toward males.
* Use Controlled Sampling to randomly oversample and adjust the training set’s gender ratio to the target proportion of approximately 55% male and 45% female.

### Implementation

* Use a custom function `p2data` to convert the DataFrame into a list, group by gender and income label, balance positive and negative samples, and perform random sampling.
* After intervention, the training set’s gender distribution becomes:

  * Female: 19,184
  * Male: 5,929

### Model Performance

* Retrained model results:

  * **Accuracy**: 0.8087
  * **Confusion Matrix**:

    ```
    [[10064  2371]
     [  743  3103]]
    ```
  * **TPR**:

    * Male TPR: 0.8025
    * Female TPR: 0.8305
* Although overall accuracy decreased, TPR improved significantly for both genders, enhancing fairness.

---

## 5. Dynamic Weight Adjustment

### Concept

* In addition to adjusting data distribution, assign different weights to samples during model training to mitigate imbalance.
* Based on gender (using a new `z` column where Female = 0 and Male = 1) and the income label, dynamically compute sample weights, giving higher weight to positive examples of the disadvantaged group (females).

### Implementation and Results

* Train the `LogisticRegression` model using the `sample_weight` parameter.
* Results:

  * **Accuracy**: 0.8485
  * **Confusion Matrix**:

    ```
    [[11455   980]
     [ 1486  2360]]
    ```
  * **TPR**:

    * Male TPR: 0.6139
    * Female TPR: 0.6119
* The model achieves a good balance between overall performance and fairness.

---

## 6. Method 1 from Literature: Adjusted Reweighting with Smoothing

### Theoretical Background

* Based on the work of Kamiran and Calders (2011), the original weight formula is:
  $\text{weight} = \frac{n_{\text{sex}} \times n_{\text{income}}}{n_{\text{total}} \times n_{\text{sex,income}}}$

* To prevent over-adjustment, introduce a smoothing parameter `lambda` so that new weights transition smoothly between 1 and the original weight.

### Implementation and Results

* After computing the original weights, apply smoothing with `lambda_param` (e.g., 0.3).
* Results:

  * Adjusted average weight for both genders is 1
  * **Accuracy**: 0.8509
  * **Confusion Matrix**:

    ```
    [[11597   838]
     [ 1590  2256]]
    ```
  * **TPR**:

    * Male TPR: 0.5866
    * Female TPR: 0.5864

---

## 7. Method 2 from Literature: Massaging (Label Modification)

### Method Principle

* Flip the labels of samples near the decision boundary to reduce the positive rate gap between genders while preserving overall model performance.
* Flip high-scoring negative samples from the disadvantaged group (females) and low-scoring positive samples from the advantaged group (males).

### Steps and Implementation

* Use Logistic Regression to compute the positive class probability score for each sample.
* Compute the number of samples to flip, `delta` (originally 2,741 in this example), then apply parameter `alpha` (e.g., 0.068) to adjust the actual number of flips (186 in this example).
* Retrain the model after flipping. Results:

  * **Accuracy**: 0.8506
  * **Confusion Matrix**:

    ```
    [[11495   940]
     [ 1493  2353]]
    ```
  * **TPR**:

    * Male TPR: 0.6121
    * Female TPR: 0.6102

---

## 8. Overall Comparison and Discussion

### Results Comparison Table

| Method                    | Accuracy | Male TPR | Female TPR |
| ------------------------- | -------- | -------- | ---------- |
| Baseline                  | 0.8530   | 0.6118   | 0.5254     |
| Controlled Resampling     | 0.8087   | 0.8025   | 0.8305     |
| Dynamic Weight Adjustment | 0.8485   | 0.6139   | 0.6119     |
| Adjusted Reweighting      | 0.8509   | 0.5866   | 0.5864     |
| Massaging                 | 0.8506   | 0.6121   | 0.6102     |

### Discussion

* **Baseline Model**: High overall accuracy, but female TPR is significantly lower than male TPR, indicating bias.
* **Controlled Resampling**: Greatly improves TPR for both genders, though accuracy decreases.
* **Dynamic Weight Adjustment & Massaging**: Achieve a balance between high accuracy and balanced TPR across genders.
* **Adjusted Reweighting**: Accuracy approaches the baseline but TPR slightly decreases.
