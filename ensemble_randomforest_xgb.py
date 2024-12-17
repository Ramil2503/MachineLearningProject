from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Define the models using best parameters
random_forest = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
xgboost = XGBRegressor(learning_rate=0.2, n_estimators=100, max_depth=6, random_state=42)

# Create the ensemble model using MultiOutputRegressor
ensemble_model = MultiOutputRegressor(
    VotingRegressor(
        estimators=[
            ('Random Forest', random_forest),
            ('XGBoost', xgboost)
        ]
    )
)

# Build the pipeline: preprocessing -> ensemble model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ensemble_model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred_ensemble = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print("\nEnsemble Model Results:")
print(f"Mean Absolute Error (MAE): {mae_ensemble}")
print(f"Mean Squared Error (MSE): {mse_ensemble}")
print(f"R² Score: {r2_ensemble}")
