# Thesis-Predicting-Moral-Preferences-in-AV-in-AU
# README for Moral Machine Dataset Analysis

## Overview

This repository contains a Jupyter Notebook that explores, manipulates, and analyzes the Moral Machine dataset. The primary focus is on data from African countries to answer the research questions:This research aims to predict and explore moral preferences across the regions of the African Union, particularly moral decision making regarding characters representing ethical socioeconomic issues, by comparing the performance of supervised machine learning models resulting from the state of the art. The analysis includes data loading, filtering, exploratory data analysis (EDA), feature engineering, and correlation analysis to understand the moral decision-making patterns in autonomous vehicle scenarios before continuing to training and testing the model.

## Dataset

The Moral Machine dataset, available at [https://osf.io/wt6mc](https://osf.io/wt6mc), contains responses from participants worldwide on various moral dilemmas involving autonomous vehicles. This project specifically focuses on responses from African countries.

## Project Structure

- **Establishing Dataset**: Loading, filtering, and saving a subset of the dataset containing only African countries.
- **Exploratory Data Analysis (EDA)**: Initial analysis to understand the representation and distribution of the dataset.
- **Feature Engineering**: Creating new features to simplify and enhance the analysis.
- **Correlation Analysis**: Investigating relationships between different features and the decision outcomes.
- **Training and Testing procedure**

## Libraries and Packages

The following libraries and packages are used in the analysis:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from scipy.stats import uniform
import seaborn as sns

```

## Data Loading and Filtering

The dataset is loaded in chunks due to its large size, and then filtered to include only responses from African countries. The filtered dataset is saved to a new CSV file for further analysis.

## Exploratory Data Analysis (EDA)

The EDA includes checking the size of the filtered dataset, identifying unique countries represented, and analyzing the distribution of responses. This step ensures that the dataset is adequately balanced and represented.

## Feature Engineering

New features are created to aggregate related character types into broader categories. This simplifies the analysis and allows for a clearer understanding of the data.

## Correlation Analysis

Correlation analysis is conducted to understand the relationship between different features (e.g., character types) and the decision outcomes (saved or killed). Both point-biserial and Pearson correlation coefficients are calculated to provide insights into these relationships.

## Usage

To run the analysis, open the provided Jupyter Notebook in your preferred environment and follow the steps outlined in the notebook. Ensure that you have the required libraries installed.

1. **Clone the Repository**: `git clone <repository_url>`
2. **Navigate to the Directory**: `cd <repository_directory>`
3. **Open the Notebook**: Use Jupyter Lab, Jupyter Notebook, or any compatible environment to open `moral_machine_analysis.ipynb`.
4. **Run the Notebook**: Execute the cells sequentially to reproduce the analysis.

### Features

1. **RegionAU_central**: Dummy variable indicating if the instance belongs to the central region of Africa.
2. **RegionAU_northern**: Dummy variable indicating if the instance belongs to the northern region of Africa.
3. **RegionAU_eastern**: Dummy variable indicating if the instance belongs to the eastern region of Africa.
4. **RegionAU_western**: Dummy variable indicating if the instance belongs to the western region of Africa.
5. **RegionAU_southern**: Dummy variable indicating if the instance belongs to the southern region of Africa.
6. **Adult**: Indicates the presence of an adult in the instance.
7. **Pregnant_2**: Indicates the presence of a pregnant individual in the instance.
8. **Stroller_2**: Indicates the presence of a stroller in the instance.
9. **Elderly**: Indicates the presence of an elderly person in the instance.
10. **Child**: Indicates the presence of a child in the instance.
11. **Homeless_2**: Indicates the presence of a homeless individual in the instance.
12. **Larger Adult**: Indicates the presence of a larger adult in the instance.
13. **Criminal_2**: Indicates the presence of a criminal in the instance.
14. **Executive**: Indicates the presence of an executive in the instance.
15. **Fit Adult**: Indicates the presence of a fit adult in the instance.
16. **Doctor**: Indicates the presence of a doctor in the instance.
17. **Animal**: Indicates the presence of an animal in the instance.
18. **2024**: This seems to be a placeholder or a specific feature related to the year 2024.

### Target Variable

- **Saved**: The target variable indicating whether the individual(s) in the instance were saved (1) or not saved (0).

### Explanation of Features

- **RegionAU_***: These dummy variables indicate the geographical region within Africa. They help the models understand the potential regional differences in moral decisions and GDP impact.
- **Adult, Pregnant_2, Stroller_2, Elderly, Child, Homeless_2, Larger Adult, Criminal_2, Executive, Fit Adult, Doctor, Animal**: These features describe the characteristics of the individuals involved in the moral dilemma. For example, the presence of a child or an elderly person could influence the decision on who to save.
- **2024**: This feature might represent some specific context or data relevant to the year 2024. The exact meaning would depend on the context provided by the dataset's description or documentation.

### Use in Models

The models (Decision Tree, SVM, and Complement Naive Bayes) use these features to learn patterns and make predictions about the target variable (Saved). The feature importance or coefficients in each model indicate how much each feature contributes to the prediction. 

- **Decision Tree**: The feature importance is determined by how much each feature reduces impurity (e.g., Gini impurity or entropy) in the tree.
- **SVM**: The coefficients of the linear SVM model indicate the weight of each feature in the decision function.
- **Complement Naive Bayes**: The feature importance can be interpreted from the feature log probabilities, showing how each feature contributes to the likelihood of the target class.
- 
## Key Findings

- The dataset from African countries contains 342,081 rows.
- All African countries except Eritrea are represented in the dataset.
- The most frequent country in the dataset is South Africa, while the least frequent is the Central African Republic.
- Feature engineering creates new aggregated features such as 'Adult', 'Child', 'Elderly', etc.
- Correlation analysis provides insights into the relationships between different character types and the decision outcomes.

## Visualization

The analysis includes various visualizations to illustrate the distribution and proportions of different character types saved or killed. These visualizations are created using `matplotlib` and provide a clear understanding of the data.

## Conclusion
The analysis reveals that demographic factors are significant determinants of moral decision-making, outweighing regional indicators. The models demonstrate a mean F1-score of 59\% across both classes and a mean PRAUC of 65\%. Among the models, the DT performs the best, achieving an F1-score of roughly 61\% for both the 'killed' and 'saved' classes, and a PRAUC of roughly 71\%. The findings suggest that demographic attributes play a crucial role in moral decision-making in autonomous vehicle scenarios. 

## Acknowledgements

The Moral Machine dataset is publicly available at [https://osf.io/wt6mc](https://osf.io/wt6mc). Special thanks to the creators of the dataset for making this research possible.

## Contact

For any questions or further information, please contact [Your Name] at [Your Email].

---

This README provides an overview of the project, guiding users through the analysis steps and highlighting key findings. It ensures that anyone interested in replicating the analysis or understanding the results can do so efficiently.
