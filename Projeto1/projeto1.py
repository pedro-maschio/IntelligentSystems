import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn import discriminant_analysis



training_dataset = pd.read_csv("hepatitis.data.train.csv", index_col=0)
test_dataset = pd.read_csv("hepatitis.data.test.csv", index_col=0)
full_dataset = pd.concat([training_dataset, test_dataset], axis = 0)

#print(full_dataset)

print(full_dataset.groupby('CLASS').count())

#full_dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
#plt.show()



full_dataset.replace("?", np.NaN, inplace=True)

# print(full_dataset)

missing_data = full_dataset.isna()
print(missing_data.shape)
# full_dataset.dropna(inplace=True)
# print(full_dataset.shape)

print(full_dataset)

full_dataset["STEROID"].fillna(full_dataset["STEROID"].astype('float').mean(), inplace=True)
full_dataset["FATIGUE"].fillna(full_dataset["FATIGUE"].astype('float').mean(), inplace=True)
full_dataset["MALAISE"].fillna(full_dataset["MALAISE"].astype('float').mean(), inplace=True)
full_dataset["ANOREXIA"].fillna(full_dataset["ANOREXIA"].astype('float').mean(), inplace=True)
full_dataset["LIVER_BIG"].fillna(full_dataset["LIVER_BIG"].astype('float').mean(), inplace=True)
full_dataset["LIVER_FIRM"].fillna(full_dataset["LIVER_FIRM"].astype('float').mean(), inplace=True)
full_dataset["SPLEEN_PALPABLE"].fillna(full_dataset["SPLEEN_PALPABLE"].astype('float').mean(), inplace=True)
full_dataset["SPIDERS"].fillna(full_dataset["SPIDERS"].astype('float').mean(), inplace=True)
full_dataset["ASCITES"].fillna(full_dataset["ASCITES"].astype('float').mean(), inplace=True)
full_dataset["VARICES"].fillna(full_dataset["VARICES"].astype('float').mean(), inplace=True)
full_dataset["BILIRUBIN"].fillna(full_dataset["BILIRUBIN"].astype('float').mean(), inplace=True)
full_dataset["ALK_PHOSPHATE"].fillna(full_dataset["ALK_PHOSPHATE"].astype('float').mean(), inplace=True)
full_dataset["SGOT"].fillna(full_dataset["SGOT"].astype('float').mean(), inplace=True)
full_dataset["ALBUMIN"].fillna(full_dataset["ALBUMIN"].astype('float').mean(), inplace=True)
full_dataset["PROTIME"].fillna(full_dataset["PROTIME"].astype('float').mean(), inplace=True)
print(full_dataset.isnull().sum())

print(full_dataset.head(10))
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print (missing_data[column].value_counts())
#     print("")   

full_dataset.hist()

