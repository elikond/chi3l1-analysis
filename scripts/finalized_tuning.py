import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier

def get_data(patientID):
	X = pd.read_csv('~/scRNA/X_' + patientID +'_scRNA.csv')
	y = pd.read_csv('~/scRNA/y_' + patientID +'_scRNA.csv')
	barcode_len = len(y.columns)
	return X, y, barcode_len

def tune_train(patient_list):
	final_dict = dict()
	final_scores = dict()
	for patientID in patient_list:
		final_scores[patientID] = dict()
		X, y, barcode_len = get_data(patientID)
		print(X)
		print(y)
		X_arr = X.iloc[:,1:].to_numpy()
		y_arr = y.iloc[:,2:].to_numpy()
		X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.33, random_state=42)
		print(y_train)
		print(X_train)
		
		model = MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic'))

		params = {
			'estimator__n_estimators': [i for i in range(50, 225, 25)],
			'estimator__max_depth': [2, 4, 6, 8, 10],
			'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4]
		}
		
		clf = RandomizedSearchCV(model, params, cv=5, scoring='recall_macro', return_train_score=False, error_score='raise')
		search = clf.fit(X_train, y_train)
		best_params = search.best_params_
		print(best_params)

		final_dict[patientID] = best_params

		new_model = MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic'))
		new_model.set_params(**best_params)

		new_model.fit(X_train, y_train)
		y_predict = new_model.predict(X_test)

		for i in range(barcode_len):
			precision, recall, thresholds = precision_recall_curve(y_test[:,i], y_predict[:,i])
			auprc = auc(recall, precision)
			final_scores[patientID][i] = auprc

		joblib.dump(new_model, patientID + '_model.pkl') 
		
		print('Final Tuning Dictionary', final_dict)
		print('Final AUPRC Scores', final_scores)
		
		return final_dict, final_scores

tune_train(['gb9', 'gb13', 'wcr8'])
