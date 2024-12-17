from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

# List of regression models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# Split data into train and test sets (random splitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reuse the existing preprocessor
for model_name, model in models.items():
    # Wrap the model in MultiOutputRegressor for multi-output regression
    multi_output_model = MultiOutputRegressor(model, n_jobs=-1)
    
    # Create the pipeline for each model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Use the same preprocessor as in the initial code
        ('regressor', multi_output_model)
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation results
    print(f"\n{model_name} Results:")
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R² Score: {r2}')
