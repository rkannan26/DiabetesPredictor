import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the dataset
myDF = pd.read_csv("diabetes.csv")
print(myDF.info())

'''
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
'''

# Now use Seaborn and Matplotlib to perform Violin Plot and Boxplot on both Glucose and DPF to see if ancestry has
# an indication whether a person has diabetes

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)

# Violin plot for Glucose levels of Non-Diabetic and Diabetic People

sns.violinplot(x='Outcome', y='Glucose', data=myDF)
plt.xlabel('Outcome')
plt.ylabel('Glucose Level')
plt.title('Violin Plot of Glucose Levels vs. Outcome')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])

# Boxplot plot for DPF levels of Non-Diabetic and Diabetic People

plt.subplot(2, 2, 2)
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=myDF)
plt.xlabel('Outcome')
plt.ylabel('DBF')
plt.title('Box Plot of Diabetes Pedigree Function vs. Outcome')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])

# Hist Plot for Pregnancies for Diabetic and Non-Diabetic People

plt.subplot(2, 2, 3)
sns.histplot(data=myDF, x='Pregnancies', hue='Outcome', multiple='stack', kde=True)
plt.xlabel('Pregnancies')
plt.ylabel('Number of People')
plt.title('Histogram of Pregnancies vs. Outcome')
plt.legend(title='Outcome', labels=['No Diabetes', 'Diabetes'])

# Strip Plot for BMI for Diabetic and Non-Diabetic People

plt.subplot(2, 2, 4)
sns.stripplot(x='Outcome', y='BMI', hue='Outcome', data=myDF, jitter=True, palette={0: "skyblue", 1: "coral"})
plt.xlabel('Outcome')
plt.ylabel('BMI')
plt.title('Scatter Plot of BMI by Outcome')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])

plt.show()

