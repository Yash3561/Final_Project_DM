# Banknote Authentication Model Evaluation

This project evaluates different machine learning models for classifying banknotes as authentic or counterfeit. Models like Random Forest, LSTM, KNN, and SVM are compared based on metrics such as accuracy, AUC, precision, and recall.

## Features
- Preprocessing and analysis of the banknote dataset.
- Training and evaluation of multiple models using 10-fold cross-validation.
- Visualizations:
  - ROC curves
  - Correlation heatmap
  - Pair plots
- Dynamic insights and recommendations for real-world applications.

## Prerequisites
To run this project, you need the following Python libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`, `scikeras`, `tensorflow`, and `tabulate`. 

Install all dependencies using:
Python 3.6 or higher is required.

## Dataset
Ensure the dataset `banknote_dataset.csv` is in the same directory as the main code file.

## How to Run
1. Clone this repository to your local machine
2.  Install dependencies:
   - `pip install -r requirements.txt`
3. Run the main Python script: `python source_code.py`
4. Alternatively, open the Jupyter Notebook: `jupyter notebook`
Navigate to the `.ipynb` file and execute the cells.

## Outputs
- **Metrics Table**: Displays performance metrics for each model over 10 folds.
- **Visualizations**:
  - Target variable distribution.
  - Correlation matrix heatmap.
  - ROC curves for all models.
- **Dynamic Insights**:
  - Best and worst models for each metric.
  - Recommendations for improving model performance.

## Troubleshooting
- **Missing Library Errors**: Ensure all libraries are installed.
- Use: `pip install -r requirements.txt`


