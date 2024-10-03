import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('snappad.csv')
df.drop(['URL', 'SKU', 'Notes', 'Status', 'Status.1', 'Product',], axis=1, inplace=True)
df['Pack'] = df['Pack'].astype(str)
df['jack'] = ''
df['Year'] = df['Year'].astype(int)

for i in range(len(df)):
    product_handle = df.loc[i, 'Pack']
    if product_handle and product_handle[0].isdigit():  # Check if the first character is a digit
        df.loc[i, 'jack'] = int(product_handle[0])
    else:
        df.loc[i, 'jack'] = np.nan  # Assign NaN for non-numeric first characters

df['jack'] = df['jack'].astype('Int64')  # Use Int64 dtype

df.drop(['Pack'], axis=1, inplace=True)

categorical_columns = ['RV Type', 'Manufacturer', 'Model Name', 'Trim', 'Leveling System']
df_encoded = pd.get_dummies(df, columns=categorical_columns)
x = df_encoded.drop(columns=['Product Handle'])
y = df_encoded['Product Handle']

x = x.dropna()
y = y.dropna()
x = x.loc[y.index]
y = y.loc[x.index]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(results)


input_data = {
    'RV Type': ['Class A'],  #
    'Manufacturer': ['Airstream'],
    'Model Name': ['Classic 30RBT'],
    'Trim': [''],
    'Leveling System': [''],
    'jack': [4],  
    'Year': [2023]
}


test = pd.DataFrame(input_data)


categorical_columns2 = ['RV Type', 'Manufacturer', 'Model Name', 'Trim', 'Leveling System']
oneHotEncoding = pd.get_dummies(test, columns=categorical_columns2)
oneHotEncoding = oneHotEncoding.reindex(columns=x.columns, fill_value=0)


x_values = oneHotEncoding
prediction = model.predict(x_values)
print(prediction)
