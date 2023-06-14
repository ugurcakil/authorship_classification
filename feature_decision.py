import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

# Veri setini yükleyin
data = pd.read_csv('C:/Users/ugrck/Downloads/out.csv')

X = data.drop('author', axis=1)
y = data['author']

# L1 Regularizasyonu ile özellik seçimi
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
model_l1.fit(X, y)

threshold_l1 = 0.3  # L1 Regularizasyonu için eşik değeri
feature_selector_l1 = SelectFromModel(model_l1, threshold=threshold_l1, prefit=True)
selected_features_l1 = X.columns[feature_selector_l1.get_support()]

print("L1 Regularizasyonu ile seçilen özellikler:")
print(selected_features_l1)

# Bilgi kazancı ile özellik seçimi
info_gains = mutual_info_classif(X, y, random_state=42)
threshold_info_gain = 0.10  # Bilgi kazancı için eşik değeri
selected_features_info_gain = X.columns[info_gains > threshold_info_gain]

print("\nBilgi kazancı ile seçilen özellikler:")
print(selected_features_info_gain)