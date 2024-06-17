import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('test_set_features.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of target variables
sns.countplot(data['xyz_vaccine'])
plt.title('Distribution of XYZ Vaccine')
plt.show()

sns.countplot(data['seasonal_vaccine'])
plt.title('Distribution of Seasonal Vaccine')
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
categorical_features = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 
                        'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 
                        'employment_industry', 'employment_occupation']

encoder = LabelEncoder()
for feature in categorical_features:
    data_imputed[feature] = encoder.fit_transform(data_imputed[feature])

# Scale numerical features if necessary
scaler = StandardScaler()
numerical_features = ['xyz_concern', 'xyz_knowledge', 'opinion_xyz_vacc_effective', 'opinion_xyz_risk', 
                      'opinion_xyz_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 
                      'opinion_seas_sick_from_vacc']

data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])

# Separate features and target variables
X = data_imputed.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_xyz = data_imputed['xyz_vaccine']
y_seasonal = data_imputed['seasonal_vaccine']


# Example feature engineering
data_imputed['total_household'] = data_imputed['household_adults'] + data_imputed['household_children']

# Update feature set
X = data_imputed.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Split the data into training and validation sets
X_train_xyz, X_val_xyz, y_train_xyz, y_val_xyz = train_test_split(X, y_xyz, test_size=0.2, random_state=42)
X_train_seasonal, X_val_seasonal, y_train_seasonal, y_val_seasonal = train_test_split(X, y_seasonal, test_size=0.2, random_state=42)

# Train the model for xyz_vaccine
model_xyz = RandomForestClassifier(random_state=42)
model_xyz.fit(X_train_xyz, y_train_xyz)

# Train the model for seasonal_vaccine
model_seasonal = RandomForestClassifier(random_state=42)
model_seasonal.fit(X_train_seasonal, y_train_seasonal)

# Predict probabilities
y_pred_xyz = model_xyz.predict_proba(X_val_xyz)[:, 1]
y_pred_seasonal = model_seasonal.predict_proba(X_val_seasonal)[:, 1]

# Evaluate the models
roc_auc_xyz = roc_auc_score(y_val_xyz, y_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_val_seasonal, y_pred_seasonal)
print(f'ROC AUC for xyz_vaccine: {roc_auc_xyz}')
print(f'ROC AUC for seasonal_vaccine: {roc_auc_seasonal}')
print(f'Mean ROC AUC: {(roc_auc_xyz + roc_auc_seasonal) / 2}')



# Load and preprocess the test set
test_data = pd.read_csv('test_data.csv')
test_data_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

for feature in categorical_features:
    test_data_imputed[feature] = encoder.transform(test_data_imputed[feature])

test_data_imputed[numerical_features] = scaler.transform(test_data_imputed[numerical_features])

# Update feature set for test data
X_test = test_data_imputed.drop(columns=['respondent_id'])

# Predict probabilities
test_data['xyz_vaccine'] = model_xyz.predict_proba(X_test)[:, 1]
test_data['seasonal_vaccine'] = model_seasonal.predict_proba(X_test)[:, 1]

# Prepare submission file
submission = test_data[['respondent_id', 'xyz_vaccine', 'seasonal_vaccine']]
submission.to_csv('submission.csv', index=False)
