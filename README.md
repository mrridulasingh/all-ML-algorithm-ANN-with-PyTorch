# Avocado Price prediction using ML algorithms and ANN with PyTorch

This data was downloaded from the Hass Avocado Board website in May of 2018 & compiled into a single CSV. Here's how the Hass Avocado Board describes the data on their website.

Some relevant columns in the dataset:

1. Date - The date of the observation
2. AveragePrice - the average price of a single avocado
3. type - conventional or organic
4. year - the year
5. Region - the city or region of the observation
6. Total Volume - Total number of avocados sold
7. 4046 - Total number of avocados with PLU 4046 sold
8. 4225 - Total number of avocados with PLU 4225 sold
9. 4770 - Total number of avocados with PLU 4770 sold


## Documentation

1. Uploading dataset and extracted basic information
* Avocado dataset consist of 18249 rows × 14 columns.
* After removel of outlier dataset contain 14306 rows, 14 columns. 
* model was build on filtered dataset.
* Label encoding
* Normalization

2. ML algorithm was used:
* Logistic Regression
* XGBoost: XGBoost is an efficient implementation of gradient boosting that can be used for regression predictive modeling. How to evaluate an XGBoost regression model using the best practice technique of repeated k-fold cross-validation.
* RandomForest Regressor
* CatBoost: CatBoost builds upon the theory of decision trees and gradient boosting. The main idea of boosting is to sequentially combine many weak models (a model performing slightly better than random chance) and thus through greedy search create a strong competitive predictive model. Because gradient boosting fits the decision trees sequentially, the fitted trees will learn from the mistakes of former trees and hence reduce the errors. This process of adding a new function to existing ones is continued until the selected loss function is no longer minimized.
* Bagging Regressor: * Bagging is an ensemble machine learning algorithm that combines the predictions from many decision trees. It is also easy to implement given that it has few key hyperparameters and sensible heuristics for configuring these hyperparameters. Bagging performs well in general and provides the basis for a whole field of ensemble of decision tree algorithms such as the popular random forest and extra trees ensemble algorithms, as well as the lesser-known Pasting, Random Subspaces, and Random Patches ensemble algorithms.
*  LGBMRegressor: LightGBM extends the gradient boosting algorithm by adding a type of automatic feature selection as well as focusing on boosting examples with larger gradients. This can result in a dramatic speedup of training and improved predictive performance. LightGBM uses a novel technique of Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value 
*  Support Vector Machine
* K Nearset Neighbor: KNN algorithm is by far more popularly used for classification problems. My aim here is analyze how KNN can be equally effective when the dependent feature is continuous in nature.
* Decision Tree Regressor
* Voting regressor: Voting is an ensemble machine learning algorithm. For regression, a voting ensemble involves making a prediction that is the average of multiple other regression models.  A voting ensemble works by combining the predictions from multiple models. It can be used for classification or regression. In the case of regression, this involves calculating the average of the predictions from the models. In the case of classification, the predictions for each label are summed and the label with the majority vote is predicted.

* 1. Regression Voting Ensemble: Predictions are the average of contributing models.
* 2. Classification Voting Ensemble: Predictions are the majority vote of contributing models.

* ANN with Pytorch: created ANN with Pytorch.

Note: This is the first time i used Voting Regressor and PyTorch and I am learning more and try to improve it. 


## Installation



```bash
  conda install -c conda-forge lightgbm
  pip install catboost
  conda install -c anaconda py-xgboost
  

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score

from sklearn import metrics
import csv
```

    
## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mridula-singh-b45b1674/)


  
# Hi, This is Mridula! 👋

  



  





  
