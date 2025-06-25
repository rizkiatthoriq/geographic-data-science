import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import DistanceMetric
import warnings

class PlotMatcherKNN:
    """
    A class to find k-nearest neighbors for labeled data points (project plots)
    from a pool of unlabeled data points (control plots) using a specified distance metric.

    This class is designed to support matching project plots with control plots
    based on historical Stocking Index (SI) values, as required by methodologies like VM0047.
    It provides options for distance metric (Euclidean, Mahalanobis) and data scaling.
    """

    def __init__(self, k: int = 3, id_col_labeled: str = 'PP ID', 
                 id_col_unlabeled: str = 'CP ID', metric: str = 'mahalanobis', 
                 use_scaler: bool = True):
        """
        Initializes the KNNSimilarityMatcher.

        Args:
            k (int): The number of nearest neighbors to find. Defaults to 3.
            id_col_labeled (str): The name of the column containing unique IDs for labeled data.
                                  Defaults to 'PP ID'.
            id_col_unlabeled (str): The name of the column containing unique IDs for unlabeled data.
                                    Defaults to 'CP ID'.
            metric (str): The distance metric to use ('euclidean', 'mahalanobis'). Defaults to 'mahalanobis'.
            use_scaler (bool): Whether to apply StandardScaler. Defaults to True.
        """
        self.k = k
        self.id_col_labeled = id_col_labeled
        self.id_col_unlabeled = id_col_unlabeled
        self.metric = metric
        self.use_scaler = use_scaler
        
        self.knn_model = None
        self.scaler = None
        self.historical_si_cols = None
        
        # Attributes to store original dataframes for later retrieval
        self.df_unlabeled_original = None     
        self.df_labeled_original = None       
        
        # Attributes to store processed features for model operations
        self.X_unlabeled_processed = None     
        self.X_labeled_processed = None       
        
        self.cov_matrix_for_mahalanobis = None 
        self.fitted = False # Track if fit was successful

    def _validate_inputs(self, df_labeled: pd.DataFrame, df_unlabeled: pd.DataFrame, historical_si_cols: list[str]):
        """Validates input DataFrames and column names."""
        if not all(col in df_labeled.columns for col in historical_si_cols + [self.id_col_labeled]):
            missing_labeled = [col for col in historical_si_cols + [self.id_col_labeled] if col not in df_labeled.columns]
            raise KeyError(f"Missing columns in labeled data (project plots): {missing_labeled}")
        if not all(col in df_unlabeled.columns for col in historical_si_cols + [self.id_col_unlabeled]):
            missing_unlabeled = [col for col in historical_si_cols + [self.id_col_unlabeled] if col not in df_unlabeled.columns]
            raise KeyError(f"Missing columns in unlabeled data (control plots): {missing_unlabeled}")
            
        if df_labeled.empty or df_unlabeled.empty:
            print("Warning: One or both input DataFrames are empty. Returning empty DataFrame.")
            return False 
        return True

    def _prepare_data(self, df_features: pd.DataFrame, is_unlabeled: bool):
        """Extracts features and applies scaling if configured."""
        # Ensure df_features is a DataFrame before accessing columns
        if not isinstance(df_features, pd.DataFrame):
            raise TypeError("df_features must be a pandas DataFrame.")
            
        X = df_features[self.historical_si_cols].values
        
        if self.use_scaler:
            print("Applying StandardScaler...")
            if is_unlabeled:
                self.scaler = StandardScaler() 
                processed_X = self.scaler.fit_transform(X)
                self.X_unlabeled_processed = processed_X 
                return processed_X
            else: # is_labeled_to_match
                if self.scaler is None:
                     raise RuntimeError("Scaler has not been fitted. Call fit() first.")
                processed_X = self.scaler.transform(X)
                self.X_labeled_processed = processed_X 
                return processed_X
        else:
            print("Skipping scaling.")
            return X

    def _setup_knn_model(self, X_unlabeled_scaled):
        """Initializes and fits the NearestNeighbors model."""
        print(f"Initializing NearestNeighbors model with k={self.k}, metric='{self.metric}'...")
        
        knn = None
        if self.metric == 'mahalanobis':
            try:
                # Ensure X_unlabeled_scaled is not empty and has enough dimensions
                if X_unlabeled_scaled.shape[0] < 2 or X_unlabeled_scaled.shape[1] < 2:
                    raise ValueError("Mahalanobis distance requires at least 2 samples and 2 features.")
                    
                self.cov_matrix_for_mahalanobis = np.cov(X_unlabeled_scaled, rowvar=False)
                if np.linalg.det(self.cov_matrix_for_mahalanobis) == 0:
                    print("Warning: Covariance matrix is singular. Adding regularization.")
                    self.cov_matrix_for_mahalanobis += np.eye(self.cov_matrix_for_mahalanobis.shape[0]) * 1e-6 
                
                knn = NearestNeighbors(n_neighbors=self.k, algorithm='auto', metric='mahalanobis', metric_params={'VI': self.cov_matrix_for_mahalanobis})
                print("Mahalanobis distance configured successfully.")
            except Exception as e:
                print(f"Error configuring Mahalanobis distance: {e}. Falling back to Euclidean distance.")
                self.metric = 'euclidean' 
        
        if self.metric != 'mahalanobis' or knn is None:
            if self.metric == 'mahalanobis':
                print("Using Euclidean distance as fallback.")
            else:
                print(f"Using specified metric: {self.metric}")
            knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        
        self.knn_model = knn
        self.knn_model.fit(X_unlabeled_scaled)
        print("NearestNeighbors model fitted.")

    def fit(self, df_labeled: pd.DataFrame, df_unlabeled: pd.DataFrame, historical_si_cols: list[str]):
        """
        Fits the KNN model using labeled project plots and unlabeled control plots.

        Args:
            df_labeled (pd.DataFrame): DataFrame of project plots.
            df_unlabeled (pd.DataFrame): DataFrame of potential control plots.
            historical_si_cols (list[str]): List of column names for historical SI values.
        """
        print("\n--- Fitting PlotMatcherKNN ---")
        # Store historical columns before validation as it's used in validation
        self.historical_si_cols = historical_si_cols 
        
        if not self._validate_inputs(df_labeled, df_unlabeled, historical_si_cols):
            self.fitted = False
            return
            
        self.fitted = True
        # Store ORIGINAL dataframes for later retrieval
        self.df_unlabeled_original = df_unlabeled.copy()  
        self.df_labeled_original = df_labeled.copy()    

        # Adjust k if it's larger than the number of control plots
        if self.k > len(df_unlabeled):
            print(f"Warning: k ({self.k}) is greater than the number of control plots ({len(df_unlabeled)}). Setting k = {len(df_unlabeled)}.")
            self.k = len(df_unlabeled)
        
        # Prepare and scale unlabeled data for fitting the model
        X_unlabeled_scaled = self._prepare_data(df_unlabeled[self.historical_si_cols], is_unlabeled=True)
        
        # Setup and fit the KNN model
        self._setup_knn_model(X_unlabeled_scaled)
        print("--- Fitting Complete ---")

    def transform(self, df_labeled_to_match: pd.DataFrame = None) -> pd.DataFrame:
        """
        Finds the k-nearest neighbors for the labeled data.

        If df_labeled_to_match is provided, it uses that data for matching.
        Otherwise, it uses the labeled data provided during the fit() method (self.df_labeled_original).

        Args:
            df_labeled_to_match (pd.DataFrame, optional): DataFrame of project plots to match.
                                                         If None, uses data from fit().

        Returns:
            pd.DataFrame: DataFrame containing the matches.
        """
        print("\n--- Transforming (Finding Matches) ---")
        if self.knn_model is None or not self.fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Determine which DataFrame to use for matching
        if df_labeled_to_match is None:
            if self.df_labeled_original is None:
                 raise RuntimeError("No labeled data available for matching. Call fit() first.")
            df_to_match = self.df_labeled_original
            print("Using labeled data from fit() for matching.")
        else:
            df_to_match = df_labeled_to_match
            print("Using provided labeled data for matching.")
        
        # Validate the data to be matched against the stored ORIGINAL unlabeled data
        if self.df_unlabeled_original is None or self.historical_si_cols is None:
             raise RuntimeError("fit() must be called before transform().")
             
        if not self._validate_inputs(df_to_match, self.df_unlabeled_original, self.historical_si_cols):
            return pd.DataFrame()

        # Prepare the data to be matched (project plots)
        X_to_match_scaled = self._prepare_data(df_to_match[self.historical_si_cols], is_unlabeled=False)

        # Find neighbors using the fitted KNN model
        print(f"Finding {self.k} nearest neighbors...")
        distances, indices = self.knn_model.kneighbors(X_to_match_scaled)

        # Construct the results DataFrame
        results = []
        for i in range(len(df_to_match)):
            label_id = df_to_match.iloc[i][self.id_col_labeled]
            neighbor_indices = indices[i]
            
            # <<< PERBAIKAN UTAMA DI SINI >>>
            # Akses DataFrame CONTROL PLOT ASLI menggunakan atribut class `self.df_unlabeled_original`
            nearest_neighbors_data = self.df_unlabeled_original.iloc[neighbor_indices].copy()
            
            nearest_neighbors_data['Matched_PP_ID'] = label_id
            nearest_neighbors_data['distance_metric_value'] = distances[i]
            
            results.append(nearest_neighbors_data)

        if not results:
            return pd.DataFrame()

        return pd.concat(results).reset_index(drop=True)

    def fit_transform(self, df_labeled: pd.DataFrame, df_unlabeled: pd.DataFrame, historical_si_cols: list[str]) -> pd.DataFrame:
        """
        Fits the model and then transforms the labeled data to find matches.

        Args:
            df_labeled (pd.DataFrame): DataFrame of project plots.
            df_unlabeled (pd.DataFrame): DataFrame of potential control plots.
            historical_si_cols (list[str]): List of column names for historical SI values.

        Returns:
            pd.DataFrame: DataFrame containing the matches.
        """
        print("\n--- Performing fit_transform ---")
        self.fit(df_labeled, df_unlabeled, historical_si_cols)
        if not self.fitted: # If fit failed (e.g., validation error)
            return pd.DataFrame()
        
        # For fit_transform, we match the same labeled data used for fitting
        return self.transform(df_labeled_to_match=df_labeled)