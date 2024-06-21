[<img src="images/icon.jpeg" width="100" alt="alt text">

# Student Dropout and Academic Performance Prediction

This repository contains a dataset and a comprehensive analysis framework for predicting student dropout and academic performance using various machine learning and deep learning models.

## Main Objective
The primary objective of this analysis is to explore the factors associated with student dropout, develop predictive models, and conduct statistical analyses to provide insights and predictive capabilities. The focus will be on improving prediction accuracy and providing valuable information to stakeholders, such as educational institutions and policymakers.

## Data Summary
The dataset includes socio-economic factors and academic performance. The data covers a wide range of features essential for understanding and predicting student dropout and academic success.

## About the Dataset

This dataset contains extensive information for students enrolled in various undergraduate degrees at a higher education institution. The dataset includes the following features:

### Student Information
- **Student ID**: A unique identifier assigned to each student (omitted).

### Demographic Details
- **Age at enrollment**: The age of the students at the time of enrollment.
- **Gender**: Gender of the students.
- **Marital status**: The marital status of the students.
- **International**: Whether the student is an international student.

### Socio-Economic Factors
- **Mother's qualification**: The qualification of the student's mother.
- **Father's qualification**: The qualification of the student's father.
- **Mother's occupation**: The occupation of the student's mother.
- **Father's occupation**: The occupation of the student's father.

### Academic Information
- **Course**: The course taken by the student.
- **Daytime/evening attendance**: Whether the student attends classes during the day or in the evening.
- **Previous qualification**: The qualification obtained by the student before enrolling in higher education.
- **Educational special needs**: Whether the student has any special educational needs.
- **Scholarship holder**: Whether the student is a scholarship holder.
- **Curricular units 1st sem (credited)**: The number of curricular units credited by the student in the first semester.
- **Curricular units 1st sem (enrolled)**: The number of curricular units enrolled by the student in the first semester.
- **Curricular units 1st sem (evaluations)**: The number of curricular units evaluated by the student in the first semester.
- **Curricular units 1st sem (approved)**: The number of curricular units approved by the student in the first semester.
- **Curricular units 1st sem (grade)**: The grades obtained in the first semester.
  
Using this dataset, researchers can investigate two key questions:    

- Which specific predictive factors are linked with student dropout or completion?
- How do different features interact with each other?   

For example, researchers could explore if there are any characteristics (gender, age at enrollment, etc.) that are associated with higher student success rates, as well as understand what implications poverty has for educational outcomes. By answering these questions, research insights are generated that can provide critical information for administrators on formulating strategies that promote successful degree completion among students from diverse backgrounds in their institutions.

- **Prediction of Student Retention**: This dataset can be used to develop predictive models that can identify student risk factors for dropout and take early interventions to improve student retention rates.
- **Improved Academic Performance**: By using this data, higher education institutions could better understand their students' academic progress and identify areas of improvement from both an individual and institutional perspective. This will enable them to develop targeted courses, activities, or initiatives that enhance academic performance more effectively and efficiently.

## Data Exploration and Cleaning
Initial data exploration involved checking for missing values, outliers, and inconsistencies. Feature engineering steps included scaling numerical features, one-hot encoding categorical features, and undersampling to address class imbalance.

Class distribution before resampling:
| Target  | Samples |
|---------|---------|
| Graduate| 2209    |
| Dropout | 1421    |

Class distribution after resampling:
| Target  | Samples |
|---------|---------|
| Graduate| 1421    |
| Dropout | 1421    |

## Features Selection
The SelectKBest method from the sklearn.feature_selection module is a feature selection technique that selects the top k features based on a statistical measure of their relevance to the target variable.

## Model Training and Evaluation
We used various machine learning models and conducted a grid search to find the best hyperparameters. Stratified cross-validation with 8 folds and downsampling for preprocessing were utilized to ensure balanced class distribution.

We also experimented with simpler models such as Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, and Gradient Boosting with cross-validation using 8 folds.

## Key Findings and Insights

### Classification Report - Neural Network (NN) (Top 10 Features)
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Dropout      | 0.929     | 0.843  | 0.884    | 1421    |
| Graduate     | 0.856     | 0.935  | 0.894    | 1421    |
| accuracy     |           |        | 0.889    | 2842    |
| macro avg    |           |        | 0.892    | 2842    |
| weighted avg |           |        | 0.892    | 2842    |

### Best Configuration
- **Layers**: [8, 8, 8]
- **Learning Rate**: 0.01
- **F1 Score**: 0.8942

### Comparison of F1 Scores
| Complexity | Model                   | Parameters (All Features) | Parameters (Top 10 Features) | F1 Score (All Features) | F1 Score (Top 10 Features) |
|------------|-------------------------|---------------------------|------------------------------|-------------------------|----------------------------|
| 1          | Logistic Regression     | C = 0.5                   | C = 0.1                      | 0.8943                  | 0.8930                     |
| 2          | K-Nearest Neighbors     | k = 25                    | k = 25                       | 0.8650                  | 0.8804                     |
| 3          | Support Vector Machine  | C = 0.1 (Linear)          | C = 0.1 (Linear)             | 0.8959                  | 0.8943                     |
| 4          | Random Forest           | Estimators = 500          | Estimators = 250             | 0.8819                  | 0.8768                     |
| 5          | Gradient Boosting       | Estimators = 100          | Estimators = 100             | 0.8906                  | 0.8872                     |
| 6          | Neural Network          |                           | Layers = [8, 8, 8]           |                         | 0.8942                     |


### Top 10 Features Importance (Random Forest)
| Feature                             | Importance |
|-------------------------------------|------------|
| Curricular units 2nd sem (approved) | 0.1802     |
| Curricular units 1st sem (approved) | 0.1165     |
| Curricular units 2nd sem (grade)    | 0.1074     |
| Curricular units 1st sem (grade)    | 0.0936     |
| Age at enrollment                   | 0.0495     |
| Curricular units 2nd sem (evaluations)| 0.0452     |
| Curricular units 1st sem (evaluations)| 0.0387     |
| Scholarship holder                  | 0.0334     |
| Curricular units 2nd sem (enrolled) | 0.0209     |
| Gender                              | 0.0206     |

## Conclusion
This analysis provides a robust framework for predicting student dropout and academic performance using various data features. The findings offer valuable insights into the key factors associated with student success and dropout rates and demonstrate the potential of machine learning models in supporting educational decision-making.

Future analysis could involve:
- **Incorporating Additional Data**: Including more detailed personal, demographic and socio-economic data.
- **Model Refinement**: Experimenting with other machine learning algorithms or ensemble methods.
- **Institutional Implementation**: Collaborating with educational institutions to implement the model's predictions for early interventions.

## Setup

### Scripts
- `data_preprocessing.py`: Loads and preprocesses the data, including scaling and resampling.
- `feature_selection.py`: Selects the top 10 features using `SelectKBest`.
- `simple_model_evaluation.py`: Evaluates simple models using all features and the top 10 features, saving the best F1 scores.
- `feature_importance.py`: Trains a Random Forest model and prints the feature importance.
- `layer_configurations.py`: Generates layer configurations and learning rates for the neural network.
- `neural_network.py`: Trains and evaluates a flexible neural network model using the best configurations.
- `main.py`: Main script to run all the above scripts in the correct order.

### How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/reab5555/student-dropout-prediction.git
    cd student-dropout-prediction
    ```

2. Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn tqdm matplotlib seaborn tabulate torch
    ```

3. Ensure the data file `student_dataset.csv` is in the correct path as specified in the `data_preprocessing.py` script.

4. Run the main script:
    ```bash
    python main.py
    ```

### Description of Scripts

- **data_preprocessing.py**: This script loads the data from a CSV file, separates features and target, scales numerical features, and resamples the data to address class imbalance.
- **feature_selection.py**: This script uses `SelectKBest` to select the top 10 features based on their importance and resamples the selected features.
- **simple_model_evaluation.py**: This script evaluates several simple models (Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting) using both all features and the top 10 features, performing cross-validation and saving the best F1 scores.
- **feature_importance.py**: This script trains a Random Forest model using the full dataset, prints the feature importance, and checks whether the top 10 features are used.
- **layer_configurations.py**: This script generates different layer configurations and learning rates for the neural network, saving them to a JSON file.
- **neural_network.py**: This script trains a flexible neural network model using the configurations from `layer_configurations.py`, performs grid search with cross-validation to find the best configuration, plots learning curves, confusion matrix, and ROC curve, and saves the final trained model.
- **main.py**: This script sequentially runs all the above scripts in the correct order.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
