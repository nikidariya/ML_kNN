import numpy as np
import random
import copy
import pandas as pd
from typing import NoReturn, Tuple, List

# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pd.read_csv(path_to_csv)
    
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    y = np.where(y == 'M', 1, 0)
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pd.read_csv(path_to_csv)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    n_samples = len(X)
    n_train = int(n_samples * ratio)
    
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test
    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)  
    fn = np.zeros(n_classes)
    
    for true, pred in zip(y_true, y_pred):
        true_idx = class_to_idx[true]
        pred_idx = class_to_idx[pred]
        
        if true == pred:
            tp[true_idx] += 1
        else:
            fp[pred_idx] += 1
            fn[true_idx] += 1
    
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    
    for i in range(n_classes):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
    
    accuracy = np.sum(tp) / len(y_true)
    
    return precision, recall, accuracy
    
# Task 4

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.leaf_size = leaf_size
        self.X = X
        self.n_features = X.shape[1]
        self.tree = self._build_tree(np.arange(len(X)))
    
    def _build_tree(self, indices, depth=0):
        class Node:
            def __init__(self, indices, is_leaf=False, split_dim=None, split_value=None, left=None, right=None):
                self.indices = indices
                self.is_leaf = is_leaf
                self.split_dim = split_dim
                self.split_value = split_value
                self.left = left
                self.right = right
        
        n_points = len(indices)
        
        if n_points <= self.leaf_size:
            return Node(indices=indices, is_leaf=True)
        
        split_dim = depth % self.n_features
        
        points = self.X[indices]
        values = points[:, split_dim]
        
        median_idx = np.argpartition(values, len(values) // 2)[len(values) // 2]
        split_value = values[median_idx]
        
        left_mask = points[:, split_dim] <= split_value
        right_mask = ~left_mask
        
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return Node(indices=indices, is_leaf=True)
        
        left_child = self._build_tree(left_indices, depth + 1)
        right_child = self._build_tree(right_indices, depth + 1)
        
        return Node(
            indices=None,
            is_leaf=False,
            split_dim=split_dim,
            split_value=split_value,
            left=left_child,
            right=right_child
        )
    
    def query(self, X: np.array, k: int = 1, return_distance: bool = False) -> List[List]:
        results = []
        for point in X:
            neighbors = self._query_point(point, k)
            results.append(neighbors)
        return results
    
    def _query_point(self, point, k):
        nearest = []
        
        def search(node):
            if node.is_leaf:
                points = self.X[node.indices]
                distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
                
                for idx, dist in zip(node.indices, distances):
                    if len(nearest) < k:
                        nearest.append((dist, idx))
                    else:
                        max_dist = -1
                        max_index = -1
                        for i, (d, idx_old) in enumerate(nearest):
                            if d > max_dist: 
                                max_dist = d
                                max_index = i
                        if dist < max_dist:
                            nearest[max_index] = (dist, idx)
                return
            
            split_dim = node.split_dim
            if point[split_dim] <= node.split_value:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
            
            search(first)
            
            if len(nearest) < k:
                search(second)
            else:
                current_max_dist = max(nearest, key=lambda x: x[0])[0]
                if abs(point[split_dim] - node.split_value) < current_max_dist:
                    search(second)
        
        search(self.tree)
        
        nearest.sort(key=lambda x: x[0])
        return [idx for dist, idx in nearest]
        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 40):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """        
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.kd_tree = None
        self.y_train = None
        self.classes = None
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """        
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        self.kd_tree = KDTree(X, self.leaf_size)
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
    
        neighbors_indices = self.kd_tree.query(X, k=self.n_neighbors)
        
        probabilities = []
        for point_neighbors in neighbors_indices:
            neighbor_classes = self.y_train[point_neighbors]
            class_votes = np.bincount(neighbor_classes, minlength=len(self.classes))
            prob = class_votes / len(point_neighbors)
            probabilities.append(prob)
        
        return probabilities
        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)