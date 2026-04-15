import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

# Read the dataset
dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Sem-6\AI\Project\data\friday.csv')

# Clean column names
dataset.columns = dataset.columns.str.strip()

# Replace infinities with NaN
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
dataset.dropna(inplace=True)

# Encode labels
label_encoder = LabelEncoder()
dataset['Label'] = label_encoder.fit_transform(dataset['Label'])

# Prepare features and target
X = dataset.drop('Label', axis=1)
y = dataset['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Create models directory if it doesn't exist
models_dir = r'C:\Users\Admin\Desktop\Sem-6\AI\Project\models'
os.makedirs(models_dir, exist_ok=True)

# Save model and preprocessors
model_path = os.path.join(models_dir, 'model.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')
encoder_path = os.path.join(models_dir, 'encoder.pkl')
feature_names_path = os.path.join(models_dir, 'feature_names.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

with open(feature_names_path, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print('Model trained and saved successfully!')
