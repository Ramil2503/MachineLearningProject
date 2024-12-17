# Split data into train and test sets (random splitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reuse the existing preprocessor
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Use the same preprocessor as in the initial code
    ('regressor', LinearRegression())
])

# Train and evaluate
lr_pipeline.fit(X_train, y_train)
y_pred = lr_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics and accuracy
print("Simple Linear Regression with Preprocessing Results:")
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')
print(f'Accuracy: {r2 * 100:.2f}%')

# Print predicted and actual values for some test cases along with car details
print("\nPredicted vs Actual for some test cases:")
for i in range(5):  # Print first 5 test cases
    car_details = X_test.iloc[i][['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].to_dict()
    print(f"\nCar Details: {car_details}")
    for j, column in enumerate(y_test.columns):
        print(f"Predicted {column}: {y_pred[i][j]}, Actual {column}: {y_test.iloc[i][column]}")