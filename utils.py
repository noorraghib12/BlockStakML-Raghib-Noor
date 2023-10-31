from sklearn.base import BaseEstimator,TransformerMixin
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB,GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder,LabelBinarizer

class YesNoBinarize(BaseEstimator,TransformerMixin):
    def __init__(self, keys):
        self.keys=keys
        
    def fit(self, X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy()
        for key in self.keys:
            if key in X_:
                X_[key]=X_[key].apply(lambda x:1 if x==('yes' or 'married') else 0)    
            else:
                continue
        return X_



def sigmoid(x):
    return 1/(1 + np.exp(-x))