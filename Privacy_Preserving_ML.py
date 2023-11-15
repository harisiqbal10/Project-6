import pandas as pd
import numpy as np
import ssl
import math

# Class for Decision Tree Node
class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, children=None, is_leaf=False, label=None):
      """
      Decision tree node constructor.
      Parameters:
        attribute: Attribute name for the node.
        value: Value of the attribute for the current branch.
        children: List of child nodes.
        is_leaf: Boolean indicating whether the node is a leaf.
        label: Class label if the node is a leaf.
      """
      self.attribute = attribute
      self.value = value
      self.children = children if children is not None else []
      self.is_leaf = is_leaf
      self.label = label

# Function to print the tree
def print_tree(node, depth=0):
    """
    Recursively prints the decision tree.

    Parameters:
      node: Current node in the tree.
      depth: Current depth of the node in the tree.
    """
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}Leaf: {node.label}")
    else:
        print(f"{indent}{node.attribute} (Split)")
        for child in node.children:
            print(f"{indent} -> On {node.attribute} = {child.value}:")
            print_tree(child, depth + 1)

# Function to predict the label for a given instance using the decision tree
def predict(instance, node):
    """
    Predicts the label for a given instance using the decision tree.

    Parameters:
      instance: Input instance for prediction.
      node: Current node in the decision tree.

    Returns: Predicted label.
    """
    while not node.is_leaf:
        attribute_value = instance[node.attribute]
        found = False
        for child in node.children:
            if child.value == attribute_value:
                node = child
                found = True
                break
        if not found:
            return None  # Return None if no matching child is found
    return node.label

# FIND-ENTROPY-SPLIT function
def find_entropy_split(D, a, N, epsilon):
    """
    Finds the entropy for splitting attribute 'a' in dataset 'D'.

    Parameters:
      D: Dataset.
      a: Attribute for splitting.
      N: Total number of instances in the dataset.
      epsilon: Privacy parameter.

    Returns:
      Total entropy for the split.
    """
    tot = 0
    unique_values = D[a].unique()
    for j in unique_values:
        subtree = D[D[a] == j]
        subtree_count = len(subtree) + np.random.laplace(0, 1/epsilon)
        subtree_entropy = 0
        for i in D['Class'].unique():
            count = len(subtree[subtree['Class'] == i]) + np.random.laplace(0, 1/epsilon)
            p_i = count / subtree_count
            if p_i > 0:
                subtree_entropy -= p_i * math.log(p_i)
            else:
                subtree_entropy = 0
        tot += (subtree_count / N) * subtree_entropy
    return tot

# DP-ID3 function
def dp_id3(D, A, epsilon1, d):
    """
    Privacy-preserving ID3 decision tree algorithm.

    Parameters:
      D: Dataset.
      A: List of attributes.
      epsilon1: Privacy parameter.
      d: Maximum depth.

    Returns:
      Root node of the decision tree.
    """
    N = len(D) + np.random.laplace(0, 1/epsilon1)
    f = max([len(D[a].unique()) for a in A])
    if len(A) == 0 or d == 0 or f * len(D['Class'].unique()) < math.sqrt(2) / epsilon1:
        label_count = {}
        for i in D['Class'].unique():
            label_count[i] = len(D[D['Class'] == i]) + np.random.laplace(0, 1/epsilon1)
        mode = max(label_count, key=label_count.get)
        return DecisionTreeNode(is_leaf=True, label=mode)

    epsilon2 = epsilon1 / (2 * len(A))
    G = {}
    for a in A:
        G[a] = find_entropy_split(D, a, N, epsilon2) 
    a_cap = min(G, key=G.get)

    root = DecisionTreeNode(attribute=a_cap)
    for j in D[a_cap].unique():
        subtree = D[D[a_cap] == j]
        A_new = A.copy()
        A_new.remove(a_cap)
        child_node = dp_id3(subtree, A_new, epsilon1, d - 1)
        child_node.value = j
        root.children.append(child_node)
    return root 

# Function to calculate the accuracy of the decision tree
def calculate_accuracy(original_file, predictions_file):
    """
    Calculates the accuracy of the decision tree predictions.

    Parameters:
      original_file: File containing the original dataset.
      predictions_file: File containing the predicted labels.

    Returns:
      Accuracy of the predictions.
    """
    # Load the original dataset and predictions
    original_data = pd.read_csv(original_file)
    predictions = pd.read_csv(predictions_file)

    # The class label is in the first column named 'class'
    correct_predictions = sum(original_data['Class'] == predictions['Predicted_Label'])
    total_predictions = len(predictions)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    # Load the breast cancer dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data' 
    columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    ssl._create_default_https_context = ssl._create_unverified_context
    data = pd.read_csv(url, names=columns) 
    data = data.astype({col: 'category' for col in data.columns if data[col].dtype == 'object'})
    
    data.to_csv('breast_cancer.csv', index=False)
    
    # Define the privacy budget and attributes
    epsilon = 20.0  # Adjust based on your privacy requirements
    d = 4
    epsilon1 = epsilon/(2*(d+1))
    attributes = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

    # Build the decision tree with differential privacy
    decision_tree_root = dp_id3(data, attributes, epsilon1, d)
    print_tree(decision_tree_root)

    # Predict and output the results for each instance in the dataset
    predictions = []
    for index, instance in data.iterrows():
        predicted_label = predict(instance, decision_tree_root)
        predictions.append(predicted_label)
    
    # Create a DataFrame with predictions and save to CSV
    prediction_df = pd.DataFrame(predictions, columns=['Predicted_Label'])
    prediction_df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')
    
    accuracy = calculate_accuracy('breast_cancer.csv', 'predictions.csv')
    print('Accuracy:', accuracy)

if __name__ == "__main__":
    main()
