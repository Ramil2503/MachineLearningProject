from sklearn.preprocessing import MinMaxScaler

# Preprocessing pipeline
# 1. Categorical features: One-Hot Encoding
# 2. Numerical features: Imputation and Scaling

categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
numerical_cols = ['Engine Size(L)', 'Cylinders']

# Define preprocessor for columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ]), numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])
