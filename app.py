import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data():
    # Load dataset from CSV
    data = pd.read_csv('HousingData.csv')
    X = data.drop('MEDV', axis=1)
    y = data['MEDV'].values.reshape(-1, 1)
    
    # Handle missing values (if any)
    X = X.fillna(X.mean())
    
    # Normalize numerical features
    X_normalized = (X - X.mean()) / X.std()
    X_normalized = X_normalized.values
    y = y.reshape(-1, 1)
    
    # Add bias term for linear regression
    X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(X_normalized.shape[0])
    train_idx, test_idx = indices[:int(0.8 * len(indices))], indices[int(0.8 * len(indices)):]
    
    X_train = X_normalized[train_idx]
    X_test = X_normalized[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train_with_bias = X_with_bias[train_idx]
    X_test_with_bias = X_with_bias[test_idx]
    
    return X_train, X_test, y_train, y_test, X_train_with_bias, X_test_with_bias, X.columns

# Linear Regression from scratch
class LinearRegressionCustom:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        self.weights = np.zeros((X.shape[1], 1))
        for _ in range(epochs):
            gradient = np.dot(X.T, (np.dot(X, self.weights) - y)) / X.shape[0]
            self.weights -= lr * gradient
    
    def predict(self, X):
        return np.dot(X, self.weights)

# Decision Tree for Regression from scratch
class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'value': np.mean(y)}
        
        best_feature, best_threshold, best_loss = None, None, float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if sum(left_idx) < self.min_samples_split or sum(right_idx) < self.min_samples_split:
                    continue
                left_y, right_y = y[left_idx], y[right_idx]
                loss = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / n_samples
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return {'value': np.mean(y)}
        
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left_tree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {'feature': best_feature, 'threshold': best_threshold, 
                'left': left_tree, 'right': right_tree}
    
    def predict(self, X):
        # Ensure predictions have shape (n_samples, 1)
        return np.array([self._predict_one(x, self.tree) for x in X]).reshape(-1, 1)
    
    def _predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        return self._predict_one(x, tree['right'])
    
    def feature_importance(self, X, y):
        importance = np.zeros(X.shape[1])
        self._compute_importance(self.tree, X, y, importance)
        return importance / importance.sum()
    
    def _compute_importance(self, tree, X, y, importance):
        if 'value' in tree:
            return
        feature = tree['feature']
        left_idx = X[:, feature] <= tree['threshold']
        right_idx = X[:, feature] > tree['threshold']
        n_samples = len(y)
        n_left, n_right = sum(left_idx), sum(right_idx)
        if n_left == 0 or n_right == 0:
            return
        total_var = np.var(y) * n_samples
        left_var = np.var(y[left_idx]) * n_left
        right_var = np.var(y[right_idx]) * n_right
        importance[feature] += total_var - (left_var + right_var)
        self._compute_importance(tree['left'], X[left_idx], y[left_idx], importance)
        self._compute_importance(tree['right'], X[right_idx], y[right_idx], importance)

# Random Forest from scratch
class RandomForestRegressorCustom:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressorCustom(self.max_depth, self.min_samples_split)
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[indices][:, feature_indices]
            tree.fit(X_subset, y[indices])
            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees))
        for i, (tree, feature_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, feature_indices]).flatten()
        return np.mean(predictions, axis=1).reshape(-1, 1)
    
    def feature_importance(self, X, y):
        importance = np.zeros(X.shape[1])
        for tree, feature_indices in self.trees:
            tree_imp = tree.feature_importance(X[:, feature_indices], y)
            importance[feature_indices] += tree_imp
        return importance / importance.sum()

# XGBoost from scratch (simplified)
class XGBoostRegressorCustom:
    def __init__(self, n_trees=10, max_depth=3, learning_rate=0.1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
    
    def fit(self, X, y):
        self.base_pred = np.mean(y)
        residuals = y - self.base_pred
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            residuals -= self.learning_rate * tree.predict(X)
    
    def predict(self, X):
        predictions = np.full((X.shape[0], 1), self.base_pred)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
    
    def feature_importance(self, X, y):
        importance = np.zeros(X.shape[1])
        for tree in self.trees:
            importance += tree.feature_importance(X, y)
        return importance / importance.sum()

# Evaluation metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_residual / ss_total

# Main execution
def main():
    # Load data
    X_train, X_test, y_train, y_test, X_train_with_bias, X_test_with_bias, feature_names = load_and_preprocess_data()
    
    # Initialize models
    lr = LinearRegressionCustom()
    rf = RandomForestRegressorCustom(n_trees=10, max_depth=5)
    xgb = XGBoostRegressorCustom(n_trees=10, max_depth=3, learning_rate=0.1)
    
    # Train models
    lr.fit(X_train_with_bias, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    
    # Make predictions
    lr_pred = lr.predict(X_test_with_bias)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    
    # Evaluate models
    models = {'Linear Regression': lr_pred, 'Random Forest': rf_pred, 'XGBoost': xgb_pred}
    for name, pred in models.items():
        print(f"{name} - RMSE: {rmse(y_test, pred):.4f}, R²: {r2_score(y_test, pred):.4f}")
    
    # Feature importance for tree-based models
    rf_importance = rf.feature_importance(X_train, y_train)
    xgb_importance = xgb.feature_importance(X_train, y_train)
    
    # Plot feature importance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(feature_names)), rf_importance)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.title('Random Forest Feature Importance')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(feature_names)), xgb_importance)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.title('XGBoost Feature Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == '__main__':
    main()