# Heart Disease Prediction using Machine Learning

## Project Overview
This project predicts whether a person is likely to have heart disease based on health metrics such as age, sex, blood pressure, cholesterol, chest pain type, and more.  
The model uses a **Random Forest Classifier** to learn patterns from historical patient data and predict the likelihood of heart disease for new patients.

## Dataset
- Source: [Heart Disease Dataset]( heart_disease.csv / https://www.kaggle.com/datasets/oktayrdeki/heart-disease)  


## Project Steps
1. **Load Dataset** – Read the CSV file into a pandas DataFrame.  
2. **Data Preprocessing** – Separate features (`X`) and target (`y`) and scale numerical values.  
3. **Train-Test Split** – Split dataset into training (80%) and testing (20%) sets.  
4. **Train Model** – Train a Random Forest Classifier on training data.  
5. **Evaluate Model** – Measure accuracy, confusion matrix, and classification metrics on the test set.  
6. **Make Predictions** – Predict heart disease for new patient data, including probability estimates.

## Example Output
**New Patient Prediction ---**
- Prediction (1 = Heart Disease, 0 = No Heart Disease): 1
- Prediction Probability [No Disease, Disease]: [0.34, 0.66]



