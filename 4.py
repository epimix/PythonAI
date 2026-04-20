import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv('assets/internship_candidates_final_numeric.csv')

# X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
# y = df['Accepted']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


import matplotlib.pyplot as plt
# # plt.scatter(X_test['Experience'], X_test['Grade'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# # plt.title('Prediction - Experience vs Grade')
# # plt.xlabel('Experience')
# # plt.ylabel('Grade')
# # plt.colorbar(label='predicted Accepted')
# # plt.show()

# # plt.scatter(X_test['Experience'], X_test['EnglishLevel'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# # plt.title('Prediction - Experience vs English Level')
# # plt.xlabel('Experience')
# # plt.ylabel('English Level')
# # plt.colorbar(label='predicted Accepted')
# # plt.show()

# # plt.scatter(X_test['Experience'], X_test['Age'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# # plt.title('Prediction - Experience vs Age')
# # plt.xlabel('Experience')
# # plt.ylabel('Age')
# # plt.colorbar(label='predicted Accepted')
# # plt.show()

# # plt.scatter(X_test['Experience'], X_test['EntryTestScore'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# # plt.title('Prediction - Experience vs Entry Test Score')
# # plt.xlabel('Experience')
# # plt.ylabel('Entry Test Score')
# # plt.colorbar(label='predicted Accepted')
# # plt.show()

# plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# plt.title('Prediction - English Level vs Entry Test Score')
# plt.xlabel('English Level')
# plt.ylabel('Entry Test Score')
# plt.colorbar(label='predicted Accepted')
# plt.show()


dff = pd.read_csv('assets/internship_candidates_cefr_final.csv')

# english_map = {
#     'Elementary': 1,
#     'Pre-Intermediate': 2,
#     'Intermediate': 3,
#     'Upper-Intermediate': 4,
#     'Advanced': 5
# }
# dff['EnglishLevel'] = dff['EnglishLevel'].map(english_map)

categorical_features = ['EnglishLevel']
numeric_features = ['Experience', 'Grade', 'Age', 'EntryTestScore']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

X_cefr = dff[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y_cefr = dff['Accepted']

X_cefr_train, X_cefr_test, y_cefr_train, y_cefr_test = train_test_split(
    X_cefr, y_cefr, test_size=0.2, random_state=42
)

model.fit(X_cefr_train, y_cefr_train)

y_pred = model.predict(X_cefr_test)

print("Accuracy:", accuracy_score(y_cefr_test, y_pred))

y_cefr_pred = model.predict(X_cefr_test)
plt.scatter(X_cefr_test['EnglishLevel'], X_cefr_test['EntryTestScore'], c=y_cefr_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Prediction - English Level vs Entry Test Score')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.colorbar(label='predicted Accepted')
plt.show()