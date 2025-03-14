# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Aidan Hardiman
# STUDENT EMAIL:  sc232ah@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [8 marks]: 
def load_data(file_path, delimiter=','):
    # Check file exists
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    
    try:
        num_rows, data, header_list=None, None, None
        # Load CSV
        df = pd.read_csv(file_path, delimiter=delimiter)
        # Get headers
        header_list = df.columns.tolist()
        # Retrieve data as NumPy array
        data = df.to_numpy()
        # Count rows by counting height
        num_rows = df.shape[0]

        # Return values
        return num_rows, data, header_list

    # Error opening file
    except Exception as e:
        warnings.warn(f"Error opening file: '{file_path}': {e}")
        return None, None, None

# Task 2[8 marks]: 
def filter_data(data):
    # Check if data exists
    if data is None:
        warnings.warn("Task 2: Warning - Data does not exist.")
        return None
    
    filtered_data = None
    
    # Check for rows (axis = 1) which contain an invalid value of -99
    invalid = np.any(data == -99, axis = 1)

    # Filter out invalild rows
    filtered_data = data[~invalid]

    # Return filtered data
    return filtered_data

# Task 3 [8 marks]: 
def statistics_data(data):
    # Check if data exists
    if data is None:
        warnings.warn("Task 3: Warning - Data does not exist.")
        return None
    
    coefficient_of_variation = None

    # Filter data
    fdata = filter_data(data)

    # Remove 'label' column (last column)
    values = fdata[:, :-1]

    # Calculate mean and std by column
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)

    # Compute coefficient, returning infinity where mean = 0
    coefficient_of_variation = np.where(mean_values != 0, std_values / mean_values, np.inf)
    return coefficient_of_variation

# Task 4 [8 marks]: 
def split_data(data, test_size=0.3, random_state=1):
    # Check if data exists
    if data is None:
        warnings.warn("Task 4: Warning - Data does not exist.")
        return None
    
    x_train, x_test, y_train, y_test = None, None, None, None

    # Filter data
    fdata = filter_data(data)

    # Seperate values and label (take out last column)
    x = fdata[:, :-1] # Values
    y = fdata[:, -1] # Labels

    # Split data with stratified smapling
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    np.random.seed(1)

    return x_train, x_test, y_train, y_test

# Task 5 [8 marks]: 
def train_decision_tree(x_train, y_train, ccp_alpha=0):
    model=None
    # Check if data exists
    if x_train is None or y_train is None:
        warnings.warn("Task 5: Warning - Data does not exist.")
        return None
    
    # Create decision tree
    model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    # Train decision tree on training data and training labels
    model.fit(x_train, y_train)

    # Return tree
    return model

# Task 6 [8 marks]: 
def make_predictions(model, x_test):
    y_test_predicted=None
    # Check if data exists
    if model is None or x_test is None:
        warnings.warn("Task 6: Warning - Data does not exist.")
        return None
    
    # Predict labels for test data based on training data
    y_test_predicted = model.predict(x_test)

    # Return predicted labels
    return y_test_predicted

# Task 7 [8 marks]: 
def evaluate_model(model, x, y):
    accuracy, recall=None,None
    # Check if data exists
    if model is None or x is None or y is None:
        warnings.warn("Task 7: Warning - Data does not exist.")
        return None, None
    
    # Make predictions
    py = model.predict(x)

    # Calculate accuracy and recall of predictions against actual values
    accuracy = accuracy_score(y, py)
    recall = recall_score(y, py)

    # Return success
    return accuracy, recall

# Task 8 [8 marks]: 
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp_alpha=None

    # Check if data exists
    if any(x is None for x in [x_train, y_train, x_test, y_test]):
        warnings.warn("Task 8: Warning - Data does not exist.")
        return None

    # Create and train unpruned tree
    unpruned_model = DecisionTreeClassifier(random_state=1, ccp_alpha=0)
    unpruned_model.fit(x_train, y_train)

    # Calculate unpruned accuracy
    unpruned_accuracy, _ = evaluate_model(unpruned_model, x_test, y_test)
    # Check if unpruned model has 100% accuracy, then return 0
    if np.isclose(unpruned_accuracy, 1):
        return 0
    # Allow 1% drop in accuracy
    accuracy_threshold = unpruned_accuracy - 0.01

    # Find optimal ccp_alpha value
    optimal_ccp_alpha = 0.001
    for alpha in np.arange(0.001, 1.001, 0.001): # Increment from 0.001 to 1 by 0.001 each time
        # Train decision tree
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=alpha)
        model.fit(x_train, y_train)

        # Compute accuracy
        accuracy, _ = evaluate_model(model, x_test, y_test)

        # Break when accuracy falls below threshold
        if accuracy < accuracy_threshold:
            break

        # Update optimal ccp_alpha value
        optimal_ccp_alpha = alpha

    return optimal_ccp_alpha

# Task 9 [8 marks]: 
def tree_depths(model):
    depth=None
    # Check if data exists
    if model is None:
        warnings.warn("Task 9: Warning - Data does not exist.")
        return None
    
    # Get depth of tree
    depth = model.get_depth()

    # Return depth
    return depth

 # Task 10 [8 marks]: 
def important_feature(x_train, y_train, header_list):
    best_feature=None
    # Check if data exists
    if x_train is None or y_train is None or header_list is None:
        warnings.warn("Task 10: Warning - Data does not exist.")
        return None
    
    # Iterate through ccp_alpha values until the depth is 1
    for alpha in np.arange(0, 1.01, 0.01):
        # Train decision tree
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=alpha)
        model.fit(x_train, y_train)

        # Get tree depth
        depth = model.get_depth()

        # If tree is depth 1, return most important feature
        if depth == 1:
            best_feature = header_list[model.tree_.feature[0]]
            return best_feature

        # If depth is > 1, keep track of root to return as last valid feature
        elif depth > 1:
            best_feature = header_list[model.tree_.feature[0]]

        # If depth skips 1 straight to 0, return last valid feature
        elif depth == 0 and best_feature is not None:
            return best_feature
        
    return None
    
# Task 11 [10 marks]: 
def optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list):
    best_ccp_alpha=0
    # Check if data exists
    if any(x is None for x in [x_train, y_train, x_test, y_test, header_list]):
        warnings.warn("Task 11: Warning - Data does not exist.")
        return None
    
    # Extract most important feature
    most_important_feature = important_feature(x_train, y_train, header_list)
    most_important_feature_index = header_list.index(most_important_feature)

    # Create 2D array of training data
    s_x_train = x_train[:, most_important_feature_index].reshape(-1, 1)
    s_x_test = x_test[:, most_important_feature_index].reshape(-1, 1)

    # Find optimal ccp_alpha value
    best_ccp_alpha = optimal_ccp_alpha(s_x_train, y_train, s_x_test, y_test)

    return best_ccp_alpha

# Task 12 [10 marks]: 
def optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list):
    optimal_depth=None
    # Check if data exists
    if any(x is None for x in [x_train, y_train, x_test, y_test, header_list]):
        warnings.warn("Task 12: Warning - Data does not exist.")
        return None
    
    # Extract most important feature
    most_important_feature = important_feature(x_train, y_train, header_list)
    most_important_feature_index = header_list.index(most_important_feature)

    # Remove most important feature
    remaining_features = [feature for i, feature in enumerate(header_list) if i != most_important_feature_index]

    # Find the second most important feature
    second_important_feature = important_feature(x_train[:, [i for i in range(x_train.shape[1]) if i != most_important_feature_index]], 
                                                 y_train, remaining_features)
    second_important_feature_index = header_list.index(second_important_feature)

    # Extract data for most important features
    x_train_two = x_train[:, [most_important_feature_index, second_important_feature_index]]
    x_test_two = x_test[:, [most_important_feature_index, second_important_feature_index]]

    # Find optimal ccp_alpha value
    best_ccp_alpha = optimal_ccp_alpha(x_train_two, y_train, x_test_two, y_test)

    # Make tree and find depth with given ccp_alpha value
    model = DecisionTreeClassifier(random_state=1, ccp_alpha=best_ccp_alpha)
    model.fit(x_train_two, y_train)
    optimal_depth = model.get_depth()

    # Return depth
    return optimal_depth    

# Example usage (Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}")
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train, header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
    
    # Test optimal ccp_alpha with single feature
    optimal_alpha_single = optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal ccp_alpha using single most important feature: {optimal_alpha_single:.4f}")
    print("-" * 50)
    
    # Test optimal depth with two features
    optimal_depth_two = optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal tree depth using two most important features: {optimal_depth_two}")
    print("-" * 50)        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# For example: 
# Line 80-87 is inspired by a code at https://stackoverflow.com/questions/48414212/how-to-calculate-accuracy-from-decision-trees