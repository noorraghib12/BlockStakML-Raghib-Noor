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
        self.X_=X.copy()
        return self
    
    def transform(self, X,y=None):
        for key in self.keys:
            if key in self.X_:
                self.X_[key]=self.X_[key].apply(lambda x:1 if x=='yes' else 0)    
            else:
                continue
        return self.X_



numeric_features = ["age", "balance",'duration','campaign','previous']
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_features = ["education", "job"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder()),]
)

binary_features = ['default','loan','housing',]
binary_transformer = Pipeline(
    steps=[
        ("encoder",YesNoBinarize(binary_features))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", binary_transformer, binary_features),
    ]
)



def sigmoid(x):
    return 1/(1 + np.exp(-x))