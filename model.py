# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def preprocess_data(data):
#     # Preprocess, clean, and encode data as in your code
#     # Code for transformations, e.g., `Create_list`, `Get_weather`, and `LabelEncoder`
#     # Define your feature matrix `X` and label vector `y`

#     # Scaling
#     scaler = StandardScaler()
#     X_std = scaler.fit_transform(X)

#     # Splitting data
#     X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
    
#     # Model initialization and training
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

#     # Predictions and metrics
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     return X_std, model, {'accuracy': accuracy, 'report': report}, model.best_params_
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_data(data):
    # Assume `data` is a pandas DataFrame
    # Label encoding for categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Define feature matrix X and label vector y
    # Here we assume the last column is the target variable; adjust as needed
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Last column as the target variable

    # Scaling the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
    
    # Model initialization and hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and fit it to the training data
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Return processed data, model, metrics, and best hyperparameters
    return X_std, best_model, {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix
    }, grid_search.best_params_
