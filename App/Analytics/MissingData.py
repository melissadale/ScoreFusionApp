import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.simplefilter(action='ignore', category=FutureWarning)

class Missing:
    def __init__(self, **kwargs):
        self.score_data = kwargs.get('data')
        self.modalities = kwargs.get('modalities')

        self.train = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][self.modalities]
        self.test = self.score_data[self.score_data['Train_Test'] == 'TEST'][self.modalities]

    def imputation_mean(self):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(self.train)
        self.train = imp_mean.transform(self.train)
        self.test = imp_mean.transform(self.test)

    def imputation_median(self):
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_median.fit(self.train)
        self.train = imp_median.transform(self.train)
        self.test = imp_median.transform(self.test)

    # Linear
    def imputation_bayesian(self):
        imputer = IterativeImputer(random_state=0, estimator=BayesianRidge())
        lin_reg_imputation = imputer.fit(self.train)
        self.train = lin_reg_imputation.transform(self.train)
        self.test = lin_reg_imputation.transform(self.test)

    # Non Linear
    def imputation_dtr(self):
        imputer = IterativeImputer(random_state=0, estimator=DecisionTreeRegressor())
        dt_reg_imputation = imputer.fit(self.train)
        self.train = dt_reg_imputation.transform(self.train)
        self.test = dt_reg_imputation.transform(self.test)

    # KNN
    def imputation_knn(self):
        imputer = IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=5))
        lin_reg_imputation = imputer.fit(self.train)
        self.train = lin_reg_imputation.transform(self.train)
        self.test = lin_reg_imputation.transform(self.test)

    def update_data(self):
        self.score_data[self.score_data['Train_Test'] == 'TRAIN'][self.modalities] = self.train
        self.score_data[self.score_data['Train_Test'] == 'TEST'][self.modalities] = self.test

