import shap
import pandas as pd

X = DF.drop(['Shannon'], axis = 1)

print(X.dtypes.value_counts())
print(list(X.select_dtypes(include = 'object')))

y = DF['Shannon'].ravel()

from category_encoders import OrdinalEncoder, OneHotEncoder


ordinal_encoder = OrdinalEncoder(cols = ordinal_features).fit(X)
X = ordinal_encoder.transform(X)
print(X.dtypes.value_counts())
# X
onehot_encoder = OneHotEncoder(cols = onehot_features).fit(X)
X_onehot = onehot_encoder.transform(X)

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR

# Logistic Regression
model_1_Lasso_baseline = Lasso()
model_1_Ridge_baseline = Ridge()
model_1_ElasticNet_baseline = ElasticNet()

# SVR
# model_2_SVR_baseline = SVR()

# model_2_SVR_optimal = SVR(C = 0.08493703293318676,
#                           gamma = 0.03150790201525316)
model_2_SVR_optimal= SVR(C = 7.751144284481423,
                          gamma = 0.3744553042052835)
# RandomForest
# model_3_RF_baseline = RandomForestRegressor()

model_3_RF_optimal  = RandomForestRegressor(
            n_estimators=651, # 858
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None)

# XGBoost
# model_4_XGB_baseline = XGBRegressor()

model_4_XGB_optimal  = XGBRegressor(
            booster='gbtree',
            n_estimators=585,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0,
            min_child_weight=1,
            gamma=0
        )

# LightGBM
# model_5_LGBM_baseline = lgb.LGBMRegressor()

model_5_LGBM_optimal  = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
#             metric='rmse',
            n_estimators=687,
            max_depth=-1,
            num_leaves=20,
            learning_rate=0.05795117777273521,
            reg_alpha=0.834238682395749,
            reg_lambda=0.2510193886889467
        )

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np
import time

model_evaluations = pd.DataFrame({'Model':[],
                          'k-fold':[],
                          'MSE':[],
                          'RMSE':[],
                          'MAE':[],
                          'R-squared':[],
                          'adjusted-R-squared':[],
                          'MAPE':[],
                          'training_time':[]})

kf = KFold(n_splits=5, shuffle = True, random_state=None)
i = 0
for train_index, test_index in kf.split(X_onehot, y):
    i+=1
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    X_train, y_train = X_onehot.iloc[train_index,:], y[train_index]
    X_test,  y_test  = X_onehot.iloc[test_index,:],  y[test_index]
    
    # model_1_Lasso_baseline
    start_time = time.time()
    model_1_Lasso_baseline.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_1_Lasso_baseline.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_1_Lasso_baseline",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
    # model_1_Ridge_baseline
    start_time = time.time()
    model_1_Ridge_baseline.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_1_Ridge_baseline.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_1_Ridge_baseline",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
    # model_1_ElasticNet_baseline
    start_time = time.time()
    model_1_ElasticNet_baseline.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_1_ElasticNet_baseline.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_1_ElasticNet_baseline",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
#     # model_2_SVR_baseline
#     start_time = time.time()
#     model_2_SVR_baseline.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_2_SVR_baseline.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_2_SVR_baseline",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
    # model_2_SVR_optimal
    start_time = time.time()
    model_2_SVR_optimal.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_2_SVR_optimal.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_2_SVR_optimal",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
#     # model_2_SVR_optimal2
#     start_time = time.time()
#     model_2_SVR_optimal2.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_2_SVR_optimal2.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_2_SVR_optimal2",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
#     # model_3_RF_baseline
#     start_time = time.time()
#     model_3_RF_baseline.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_3_RF_baseline.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_3_RF_baseline",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
    # model_3_RF_optimal
    start_time = time.time()
    model_3_RF_optimal.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_3_RF_optimal.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_3_RF_optimal",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
#     # model_4_XGB_baseline
#     start_time = time.time()
#     model_4_XGB_baseline.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_4_XGB_baseline.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_4_XGB_baseline",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
    # model_4_XGB_optimal
    start_time = time.time()
    model_4_XGB_optimal.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_4_XGB_optimal.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_4_XGB_optimal",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
#     # model_5_LGBM_baseline
#     start_time = time.time()
#     model_5_LGBM_baseline.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_5_LGBM_baseline.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_5_LGBM_baseline",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
    # model_5_LGBM_optimal
    start_time = time.time()
    model_5_LGBM_optimal.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model_5_LGBM_optimal.predict(X_test)
    model_evaluations.loc[len(model_evaluations)] = ["model_5_LGBM_optimal",
                                                     i,
                                                     mean_squared_error(y_test, y_pred),
                                                     np.sqrt(mean_squared_error(y_test, y_pred)),
                                                     mean_absolute_error(y_test, y_pred),
                                                     r2_score(y_test, y_pred),
                                                     1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
                                                     mean_absolute_percentage_error(y_test, y_pred),
                                                     end_time-start_time]
#     # model_5_LGBM_optimal2
#     start_time = time.time()
#     model_5_LGBM_optimal2.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_5_LGBM_optimal2.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_5_LGBM_optimal2",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
#     # model_4_XGB_optimal2
#     start_time = time.time()
#     model_4_XGB_optimal2.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_4_XGB_optimal2.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_4_XGB_optimal2",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
#      # model_3_RF_optimal2
#     start_time = time.time()
#     model_3_RF_optimal2.fit(X_train, y_train)
#     end_time = time.time()
#     y_pred = model_3_RF_optimal2.predict(X_test)
#     model_evaluations.loc[len(model_evaluations)] = ["model_3_RF_optimal2",
#                                                      i,
#                                                      mean_squared_error(y_test, y_pred),
#                                                      np.sqrt(mean_squared_error(y_test, y_pred)),
#                                                      mean_absolute_error(y_test, y_pred),
#                                                      r2_score(y_test, y_pred),
#                                                      1 - (1-r2_score(y_test, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1),
#                                                      mean_absolute_percentage_error(y_test, y_pred),
#                                                      end_time-start_time]
    
    
model_evaluations.groupby('Model').agg({'MSE': 'mean', 'RMSE': 'mean', 'MAE': 'mean', 'R-squared': 'mean', 'adjusted-R-squared': 'mean', 'MAPE': 'mean'}).sort_values('R-squared')
