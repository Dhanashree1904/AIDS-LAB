import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('diabetes.csv')
print(data.head())
print(data.info())

from ydata_profiling import ProfileReport
profile = ProfileReport(data)
profile.to_notebook_iframe()

sns.countplot(x='Outcome', data=data)
plt.title('Outcome Distribution')
plt.show()

data.hist(figsize=(10, 8))
plt.suptitle('Feature Distributions')
plt.show()

data.hist(figsize=(10, 8))
plt.suptitle('Feature Distributions')
plt.show()

print((data == 0).sum())

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['Outcome']))
scaled_df = pd.DataFrame(scaled_data, columns=data.columns[:-1])

plt.figure(figsize=(8,4))
sns.kdeplot(data['Glucose'], label='Original')
sns.kdeplot(scaled_df['Glucose'], label='Scaled')
plt.title('Effect of Scaling on Glucose Feature')
plt.legend()
plt.show()



