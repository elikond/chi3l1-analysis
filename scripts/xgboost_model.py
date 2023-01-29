import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier

supDir = '~/surprisal_analysis/'
clusterDir = '~/clusters/'
mesenDir = '~/MesenProneural/'

def convert_barcode(df, col_name):
    barcode_int_abs = list()
    barcode_int = list()
    for barcode_str in df[col_name]:
        barcode_int_abs.append([abs(int(d)) for d in re.findall(r'-?\d+', barcode_str)])
        barcode_int.append([int(d) for d in re.findall(r'-?\d+', barcode_str)])
    df['Barcode_Int'] = barcode_int
    df['Barcode_Int_Abs'] = barcode_int_abs
    
    return df

def get_processes(df, barcode_len):
    #Turning barcode strings into list of integers (and making positive)
    df = convert_barcode(df, 'barcode')
    
    #Shorten the barcodes the appropriate length
    df['Barcode_Int_Abs_Short'] = df['Barcode_Int_Abs'].apply(lambda x: x[:barcode_len])
    df['Barcode_Int_Short'] = df['Barcode_Int'].apply(lambda x: x[:barcode_len])

    #Creating a column per process
    process_columns = ["Process {}".format(i+1) for i in range(barcode_len)]
    #print(df['Barcode_Int_Short'])
    df[process_columns] = pd.DataFrame(df['Barcode_Int_Abs_Short'].tolist(), index= df.index)
    
    return df

def get_mainDF(supDir, clusterDir, mesenDir, patientID):
    #gives cluster of each cell
    clusters = pd.read_csv(clusterDir + patientID + '_clusters.txt', sep='\t')
    clusters['cell_id'] = clusters.index
    
    #scrna data
    scrnaRaw = pd.read_csv(supDir + patientID + '/' + patientID + '_sa_scRNA.csv', sep=',')
    scrna = scrnaRaw.T.rename(columns = scrnaRaw['Gene']).drop('Gene')
    scrna['cell_id'] = scrna.index.str.replace(".","-", regex=True)
    
    #gives barcode of each cell
    barcodesRaw = pd.read_csv(supDir + patientID + '/' + patientID + '_sa_lambda_barcodes.csv', sep=',')
    barcodes = barcodesRaw.rename(columns=barcodesRaw.iloc[0]).drop([0]).iloc[:,:2]
    barcodes['cell_id'] = barcodes['cell_id'].str.replace(".","-", regex=True)
    
    #gives the percent of control and treatment cells in each subpopulation/barcode
    subpop_perc = pd.read_csv(supDir + patientID + '/' + patientID + '_sa_subpopulations.csv', sep=',')
    subpop_perc = convert_barcode(subpop_perc, 'subpop_barcode')
    barcode_len = len(subpop_perc['Barcode_Int'][0])
    print(barcode_len)
    
    #loading in mesenchymal/proneural data for each cell
    mesen_proneuralRaw = pd.read_csv(mesenDir + 'mesenchymal_proneural_' + patientID + '.csv', sep=',')
    mesen_proneural = mesen_proneuralRaw[['Mesenchymal', 'Proneural']]
    mesen_proneuralCells = mesen_proneuralRaw.iloc[:,0]
    mesen_proneural['cell_id'] = mesen_proneuralCells.copy()
    
    #getting type (control or treatment) of each cell
    type_list = list()
    for cell in barcodes['cell_id']:
        if cell.endswith('_1'):
            type_list.append('control')
        else:
            type_list.append('treatment')
    barcodes['Cell_Type'] = type_list
    
    #merging all dfs into one df
    mergedTemp1 = pd.merge(clusters, barcodes, on='cell_id', how='outer')
    mergedTemp2 = pd.merge(mergedTemp1, mesen_proneural, on='cell_id', how='outer')
    merged_df = pd.merge(mergedTemp2, scrna, on='cell_id', how='outer')
    
    merged_df = get_processes(merged_df, barcode_len)
    
    X_data = merged_df.iloc[:,4:-barcode_len-4]
    y_data = merged_df.iloc[:,-barcode_len:]
    
    return merged_df, X_data, y_data, barcode_len

#cellType = control or treatment
def split_cellType(df):
    controlDF = df[df['Cell_Type'] == 'control']
    treatmentDF = df[df['Cell_Type'] == 'treatment']
    return controlDF, treatmentDF

def subpops(controlDF, treatmentDF):
    sigSubpop_DFs = list()
    for df in [controlDF, treatmentDF]:
        df1 = df.copy()
        barcodes_counts = df1['Barcode_Int_Short'].value_counts().div(len(df1)).multiply(100)
        sig_barcodes = list(barcodes_counts[barcodes_counts > 1].index)
        print(barcodes_counts)
        sigSubpop_DFs.append(df1[df1['Barcode_Int_Short'].isin(sig_barcodes)])
    new_df = pd.concat(sigSubpop_DFs)

    new_subpops = list()
    my_passed = dict()
    i = 0
    for barcode in new_df['Barcode_Int_Short']:
        if str(barcode) not in my_passed:
            new_subpops.append(i)
            my_passed[str(barcode)] = i
            i += 1
        else:
            new_subpops.append(my_passed[str(barcode)])
    new_df['Subpopulations'] = new_subpops
    
    new_df.reset_index(inplace = True, drop = True)
    
    return new_df

def final(supDir, clusterDir, mesenDir, patientID):
    merged_df, X_data, y_data, barcode_len = get_mainDF(supDir, clusterDir, mesenDir, patientID)
    controlDF, treatmentDF = split_cellType(merged_df)
    sigSubpopsDF = subpops(controlDF, treatmentDF)
    return merged_df, X_data, y_data, barcode_len, sigSubpopsDF

patientID = 'gb9'
merged_df, X_data, y_data, barcode_len, sigSubpopsDF = final(supDir, clusterDir, mesenDir, patientID)

X_data_arr = X_data.to_numpy()
y_data_arr = y_data.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_data_arr, y_data_arr, test_size=0.33, random_state=42)
print(X_train.shape)
print(y_train)

pipe = Pipeline(
    steps = [
        ('model', MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic')))
    ]
)

search = RandomizedSearchCV(
    estimator = pipe,
    param_distributions={'model__estimator__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4], 'model__estimator__n_estimators': [i for i in range(50, 225, 25)]},
    scoring = ['accuracy', 'precision', 'recall'],
    refit = 'precision',
    cv = 5
).fit(X_train, y_train)

print(search.best_params_)
