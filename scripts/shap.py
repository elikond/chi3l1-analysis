import shap
shap.initjs()
import torch
from torch.utils.data import Dataset
import numpy as np
import joblib

class ModelDataset(Dataset):
    def __init__(self, x, y, cells_df):
        self.X = torch.tensor(x.to_numpy(dtype='float32'))
        self.Y = torch.tensor(list(y['barcode']))
        self.Xdf = x
        self.Ydf = y
        self.cells = cells_df.loc[x.index,:]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return len(self.X)

def get_data(patientID):
        model_data = torch.load('/gpfs/data/rsingh47/Chi3l1_Data/preprocessed_data/' + patientID + '/' + patientID + '_model_data__inputs_scRNA_sa_nomt__labels_barcodes.pt')
        train_data = model_data['train_data']
        test_data = model_data['test_data']

        X_train = np.array(train_data.X)
        X_temp = train_data.Xdf
        features = list(X_temp.columns)
        y_train = np.array(train_data.Y)
        X_test = np.array(test_data.X)
        y_test = np.array(test_data.Y)
        barcode_len = y.shape[1]

        return X_train, y_train, X_test, y_test, features, barcode_len

patientID = 'gb2'
X_train, y_train, X_test, y_test, features, barcode_len = get_data(patientID)

tuned_model = joblib.load(patientID + '_final_model.pkl')
explainer = shap.KernelExplainer(model = model.predict, data = X_train, link = "identity")        

print('Current Label Shown: Process 1')

print(shap.summary_plot(shap_values = shap_values[0], features = features)
