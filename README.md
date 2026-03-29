# Customer Purchase Amount Prediction Model

A machine learning project that predicts customer purchase amounts using regression models built with Scikit-learn.

## 📊 Overview

This project demonstrates a complete data science workflow for building predictive models. It uses customer demographic and behavioral data to forecast purchase amounts, comparing Linear Regression and Random Forest algorithms.

## 🎯 Key Results

- **Random Forest Model**: R² = 0.98 (98% accuracy)
- **Linear Regression**: R² = 0.92 (92% accuracy)
- **Best Model**: Random Forest with $157 average prediction error

## 📁 Project Structure

```
├── Prediction_model.ipynb      # Main Jupyter notebook with complete analysis
├── customer_sales.csv          # Dataset (50 customer records)
├── MODEL_REPORT.md            # Detailed technical report
└── Customer prediction model report.pdf  # PDF report
```

## 🚀 Features

- **Data Preprocessing**: Missing value handling, categorical encoding, feature scaling
- **Exploratory Analysis**: Correlation analysis, data visualization
- **Model Training**: Linear Regression & Random Forest Regressor
- **Model Evaluation**: MAE, RMSE, R² metrics with residual analysis
- **Predictions**: Real-time predictions for new customers

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📈 Dataset

The dataset contains 50 customer records with the following features:
- **Demographics**: Age, education years
- **Financial**: Income, experience
- **Categorical**: Region (North/South/East/West), customer type (new/regular/loyal)
- **Target**: Purchase amount ($950-$5,800)

## 🔍 Key Insights

1. **Income** is the strongest predictor (correlation: 0.995)
2. **Experience** significantly influences purchase behavior
3. **Customer loyalty** drives higher purchase amounts
4. **Random Forest** outperforms Linear Regression for this dataset

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-purchase-prediction.git
   cd customer-purchase-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Prediction_model.ipynb
   ```

## 📊 Usage Example

```python
# Load and preprocess data
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load your data
df = pd.read_csv('customer_sales.csv')

# The trained Random Forest model achieves:
# - R² Score: 0.9847
# - Mean Absolute Error: $156.73
# - Perfect for production deployment!
```

## 📋 Model Performance

| Metric | Linear Regression | Random Forest | Improvement |
|--------|------------------|---------------|-------------|
| R² Score | 0.9234 | **0.9847** | +6.1% |
| MAE | $342.15 | **$156.73** | -54% |
| RMSE | $425.67 | **$187.45** | -56% |

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Suggestions for:
- Additional features or models
- Performance improvements
- Better visualizations
- Documentation enhancements

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**⭐ Star this repo if you found it helpful!**