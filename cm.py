# import statements 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Filename of the dataset to use for training and validation
train_data = "Mobile Price_train.csv"
# Filename of test dataset to apply your model and predict outcomes 
test_data = "Mobile Price_test.csv"
df = pd.read_csv(train_data)

X = df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'ram', 'talk_time']]
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('std_scaler', StandardScaler()),
    ('lr',LogisticRegression(penalty='none', max_iter=1000))
    ])
pipeline.fit(X_train,y_train)
y_predict = pipeline.predict(X_test)
training_accuracy = accuracy_score(y_test, y_predict).round(4)
confusionmatrix = confusion_matrix(y_test, y_predict)
log_reg = LogisticRegression()
disp = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix,display_labels=pipeline.classes_)
disp.plot()
plt.savefig('cm.png')
plt.show()