import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        test_length = X.shape[0]
        train_length = self.train_X.shape[0]
        
        distances = np.zeros((test_length,train_length))
        
        for i in range(test_length):
            for j in range(train_length):
                
                distances[i][j] = np.sum(np.abs(X[i] - self.train_X[j]))
                
        distances = np.array(distances)
        
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        train_length = self.train_X.shape[0]
        test_length = X.shape[0]
        
        distances = np.zeros((test_length,train_length), np.float32)
        
        for i in range(test_length):
            
            distances[i] = np.sum(np.abs(X[i] - self.train_X), 1)
            
        distances = np.array(distances)
        
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = np.sum(np.abs(X[: , None, :] - self.train_X[None, : , :]), -1)
        
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        
        el_idx = np.argsort(distances, axis=1)[:, :self.k]

        prediction = []

        for i in el_idx:
    
            if (self.train_y[i]=='0').sum() < (self.train_y[i]=='1').sum():

                prediction.append('1')
        
            else:
                prediction.append('0')
                
        prediction = np.array(prediction)

        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        min_el_idx = np.argsort(distances, axis=1)[:, :self.k]

        idx = []

        for i in min_el_idx:
    
            idx.append(np.bincount(i).argmax())
    
        prediction = np.array(self.train_y[idx])

        return prediction