from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import numpy as np

warnings.simplefilter("ignore", category=UserWarning)

# List of regression models with hyperparameter grids to tune
models = {
    "Linear Regression": {
        'model': LinearRegression(),
        'params': {}  # LinearRegression doesn't require tuning
    },
    "Random Forest": {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__estimator__n_estimators': [50, 100],
            'regressor__estimator__max_depth': [None, 10, 20],
            'regressor__estimator__min_samples_split': [2, 5]
        }
    },
    "K-Nearest Neighbors": {
        'model': KNeighborsRegressor(),
        'params': {
            'regressor__estimator__n_neighbors': [3, 5],
            'regressor__estimator__weights': ['uniform', 'distance'],
            'regressor__estimator__algorithm': ['brute']  # Force brute force search
        }
    },
    "Decision Tree": {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'regressor__estimator__max_depth': [None, 10, 20],
            'regressor__estimator__min_samples_split': [2, 5, 10],
            'regressor__estimator__min_samples_leaf': [1, 2, 4]
        }
    },
    "XGBoost": {
        'model': XGBRegressor(random_state=42),
        'params': {
            'regressor__estimator__n_estimators': [50, 100],
            'regressor__estimator__learning_rate': [0.1, 0.2],
            'regressor__estimator__max_depth': [3, 6]
        }
    }
}

# Split data into train and test sets (random splitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate through each model, apply grid search for hyperparameter tuning, and evaluate
best_models = {}
for model_name, model_info in models.items():
    model = model_info['model']
    params = model_info['params']
    
    # Wrap the model in MultiOutputRegressor for multi-output regression
    multi_output_model = MultiOutputRegressor(model, n_jobs=-1)
    
    # Create the pipeline for each model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Use the same preprocessor as in the initial code
        ('regressor', multi_output_model)
    ])
    
    # Apply GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model after tuning
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Make predictions using the best model
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation results
    print(f"\n{model_name} Results after Hyperparameter Tuning:")
    print(f"Best Params: {grid_search.best_params_}")
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R² Score: {r2}')
    
# Now, you can train the model with the best hyperparameters
# Choose the best model (lowest error or highest R²)
best_model_name = max(best_models, key=lambda k: best_models[k].score(X_test, y_test))

best_model_final = best_models[best_model_name]
best_model_final.fit(X_train, y_train)

# Make predictions
y_pred_final = best_model_final.predict(X_test)

# Final evaluation
mae_final = mean_absolute_error(y_test, y_pred_final)
mse_final = mean_squared_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

# Print final evaluation results
print(f"\nFinal Results for the Best Model ({best_model_name}):")
print(f'Mean Absolute Error (MAE): {mae_final}')
print(f'Mean Squared Error (MSE): {mse_final}')
print(f'R² Score: {r2_final}')
