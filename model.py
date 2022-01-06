import pandas as pd
from sklearn import linear_model,model_selection,metrics,ensemble
import pickle 

df = pd.read_csv("diabetes_pima.csv")
print("Data read")
#print(sorted(df.corr()['Outcome']))

x = df[['Glucose','Pregnancies','Insulin','BMI','Age']].values
y = df['Outcome'].values
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(x,y,test_size = 0.2,stratify = y)

'''sc = StandardScaler()
trans_xtrain = sc.fit_transform(xtrain)
trans_xtest = sc.transform(xtest)'''

lr = linear_model.LogisticRegression()
lr.fit(xtrain,ytrain)

rf = ensemble.RandomForestClassifier()
print("Training started for rf...")
rf.fit(xtrain,ytrain)
print("model fitted")

print('classification report \n', metrics.classification_report(ytest,lr.predict(xtest)))
print('classification report \n', metrics.classification_report(ytest,rf.predict(xtest)))


filename = "saved_model.pkl"
saved_model = pickle.dump(lr,open(filename,'wb'))
print("model saved")