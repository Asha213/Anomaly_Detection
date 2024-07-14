def process(trainpath,testpath,nbc):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    COLOR_RED = "\033[31m"
    COLOR_GREEN = "\033[32m"
    COLOR_CYAN = "\033[36m"

    def textf(text, format):
        return f"{format}{text}{RESET}"

    def bold(text):
        return textf(text, BOLD)

    def underline(text):
        return textf(text, UNDERLINE)
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
    import pickle
    def load_data() -> tuple:

        # Loading the data as indicated in the assignment
        # kddcup.data.corrected is the training data, corrected is the testing data
        print('Loading the data...', end=' ')
        training_data = pd.read_csv( trainpath, header=None )
        testing_data = pd.read_csv( testpath, header=None )
        print( textf('Done!', COLOR_GREEN) )

        # Separating the labels (last column) from the features in the training and testing data
        print('Separating the labels from the features...', end=' ')
        X_train = training_data.iloc[:, :-1]
        y_train = training_data.iloc[:, -1]

        X_test = testing_data.iloc[:, :-1]
        y_test = testing_data.iloc[:, -1]
        print( textf('Done!', COLOR_GREEN) )

        # Concatenating the training and testing data vertically to perform OneHotEncoding on all the categorical features
        print('Concatenating the training and testing data...', end=' ')
        X = pd.concat( [X_train, X_test], axis=0 )
        print( textf('Done!', COLOR_GREEN) )

        # Encoding the features into one-hot vectors using OneHotEncoder
        # ColumnTransformer is used to apply the OneHotEncoder to the second, third and fourth columns (categorical features)
        # The remainder is set to 'passthrough' to keep the other columns unchanged (already numerical)
        print('Encoding the features into one-hot vectors...', end=' ')
        ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough' )
        ct = ct.fit(X)
        X_train = pd.DataFrame( ct.transform(X_train) )
        X_test = pd.DataFrame( ct.transform(X_test) )
        print( textf('Done!', COLOR_GREEN) )

        # Feature Scaling since some features have a much higher range than others
        print('Feature Scaling...', end=' ')
        scaler = StandardScaler()
        X_train = pd.DataFrame( scaler.fit_transform( X_train ) )
        X_test = pd.DataFrame( scaler.transform( X_test ) )
        print( textf('Done!', COLOR_GREEN) )

        # Equating labels of the testing data to the training data
        print('Equating labels of the testing data to the training data...', end=' ')
        y_test = pd.Series( y_test[y_test.isin(y_train)] )
        X_test = pd.DataFrame( X_test[X_test.index.isin(y_test.index)] )
        print( textf('Done!', COLOR_GREEN) )

        print('All done!')

        return X_train, y_train, X_test, y_test
    # Randomly picking some samples from the training data to speed up the training process (for testing purposes only)
    def sample_data(X_train, y_train, X_test, y_test, n_train:int=10000, n_test:int=1000) -> tuple:
        print('Sampling the data...', end=' ')
        X_train = X_train.sample(n_train, random_state= 42)
        y_train = y_train[X_train.index]

        X_test = X_test.sample(n_test, random_state= 42)
        y_test = y_test[X_test.index]
        print( textf('Done!', COLOR_GREEN) )
        return X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = load_data()
    try:
        print('Number of unique labels in the training data: ', len(y_train.unique()))
        print('-' * 50)
        print('Number of unique labels in the testing data : ', len(y_test.unique()))
    except NameError:
        print( textf('NameError: y_train or y_test is not defined', COLOR_RED) )
    path = 'checkpoints'

    # This function saves the kmeans model state using pickle
    def save_state_kmeans( k: int, centroids: pd.DataFrame, clusters: dict, cluster_indices: dict, cluster_labels: dict ):
        with open( path + '/centroids' + str(k) + '.pkl', 'wb') as file:
            pickle.dump(centroids, file)
        with open( path + '/clusters' + str(k) + '.pkl', 'wb') as file:
            pickle.dump(clusters, file)
        with open( path + '/cluster_indices' + str(k) + '.pkl', 'wb') as file:
            pickle.dump(cluster_indices, file)
        with open( path + '/cluster_labels' + str(k) + '.pkl', 'wb') as file:
            pickle.dump(cluster_labels, file)

    # This function loads the kmeans model state using pickle
    def load_state_kmeans( k: int ) -> tuple:
        with open( path + '/centroids' + str(k) + '.pkl', 'rb') as file:
            centroids = pd.DataFrame( pickle.load(file) )
        with open( path + '/clusters' + str(k) + '.pkl', 'rb') as file:
            clusters = dict( pickle.load(file) )
        with open( path + '/cluster_indices' + str(k) + '.pkl', 'rb') as file:
            cluster_indices = dict( pickle.load(file) )
        with open( path + '/cluster_labels' + str(k) + '.pkl', 'rb') as file:
            cluster_labels = dict( pickle.load(file) )
        return centroids, clusters, cluster_indices, cluster_labels
    def kmeans_clustering( k, data, max_iterations:int=None, print_updates=False, initial_centroids=None, save_cp=True ):
        
        # Initially, selecting k random data points as centroids
        # We will use the current time as the seed to make sure that we get different centroids each time we run the algorithm
        
        np.random.seed( 42 )

        if initial_centroids is None:
            centroids = data.sample( k, random_state= 42 )
        else:
            centroids = initial_centroids.copy()
            
        old_centroids = None

        # If the user doesn't specify the maximum number of iterations, we will set it to infinity (Loop until convergence)
        if max_iterations is None:
            max_iterations = np.inf

        itr = 1
        while( itr <= max_iterations ):

            # If the centroids do not change, we will stop the algorithm
            if centroids.equals( old_centroids ):
                break

            # Storing the old centroids to check if they change in the next iteration
            old_centroids = centroids.copy()

            if print_updates is True: print( underline(bold(' Iteration #' + str(itr) + ' ')) )

            # Container initialization for cluster data
            clusters = {} # clusters[i] will store the data points in the (i+1)th cluster
            cluster_indices = {} # cluster_indices[i] will store the indices of the data points in the (i+1)th cluster
            cluster_labels = {} # cluster_labels[i] will store the label of the (i+1)th cluster by majority voting
            for i in range(k):
                clusters[i] = []
                cluster_indices[i] = []

            # Broadcasting the centroids and the data points to make the calculations easier
            # centroids is a (k x d) matrix, data is a (n x d) matrix
            # So, we will broadcast them both to a (n x k x d) matrix in order to be able to calculate the distances between them
            centroids_broadcasted = np.broadcast_to( centroids.to_numpy(), (data.shape[0], k, data.shape[1]) )
            data_broadcasted = np.broadcast_to( data.to_numpy()[:, np.newaxis, :], (data.shape[0], k, data.shape[1]) )

            # Calculating the distances between the data points and the centroids
            # We use axis=2 because we want to calculate the distance between each data point and each centroid along the feature axis
            distances = np.linalg.norm( data_broadcasted - centroids_broadcasted, axis=2 )

            # Finding the closest centroid for each data point
            # We use axis=1 because we want to find the closest centroid for each data point along the centroid axis
            closest_cluster_indices = np.argmin( distances, axis=1 )


            # Assigning the data points to the clusters
            for i in range(k):
                clusters[i] = data.iloc[ closest_cluster_indices == i ]
                cluster_indices[i] = data.index[ closest_cluster_indices == i ].tolist()

            # Updating the centroids.
            # We will use our calculate_mean function to calculate the mean of the data points in the cluster
            # because it handles both numerical and categorical data
            for i in range(k):

                # If the cluster is empty, we will not update the centroid
                if len(clusters[i]) == 0:
                    continue
                else:
                    centroids.iloc[i] = np.mean( clusters[i], axis=0 )

            
            # Calculating the cluster labels by majority voting (if the cluster is not empty)
            for i in range(k):
                if len(clusters[i]) == 0:
                    cluster_labels[i] = None
                else:
                    cluster_labels[i] = pd.Series( [y_train[index] for index in cluster_indices[i]] ).value_counts().index[0]

            # Printing the cluster sizes and labels in a pandas table
            if print_updates is True:
                print( pd.DataFrame( [ [len(clusters[i]), cluster_labels[i]] for i in range(k) ], columns=['Cluster Size', 'Cluster Label'] ) )

            if print_updates is True: print('-' * 50) # Just to print a line to separate the iterations

            if save_cp: save_state_kmeans( k, centroids, clusters, cluster_indices, cluster_labels )

            itr += 1

        return centroids, clusters, cluster_indices, cluster_labels
    # This function will be used to calculate the purity of the clusters
    # Purity is the percentage of data points in a cluster that belong to the same class
    def calculate_purity( clusters, labels, print_report=False ):
        purities = []
        for i in range( len(clusters) ):
            cluster = clusters[i]

            # If the cluster is empty, we will skip it
            if len(cluster) == 0: continue

            # Converting the cluster to a dataframe so that we can use the value_counts() function
            cluster = pd.DataFrame(cluster)
            cluster['label'] = labels[cluster.index]

            # We will use the value_counts() function to count the number of data points in each class
            # and then we will divide it by the total number of data points in the cluster
            purities.append( cluster['label'].value_counts()[0] / len(cluster) )

        # Normalizing the purity by dividing it by the number of clusters
        average_purity = sum(purities) / len(clusters)

        if print_report is True:
            for i in range(len(purities)):
                print('Cluster ', i+1, ' purity: ', purities[i])
            print('-'*50)
            print('Average Purity: ', average_purity)
            print('-'*50)
        
        return average_purity, purities


    # This function prints a report of the clusters produced by the k-means algorithm
    def analyze_clusters( clusters, cluster_indices, cluster_labels, labels ):

        # Printing the number of data points in each cluster
        for i in range(len(cluster_indices)):
            print('Cluster ', i+1, ' contains ', len(cluster_indices[i]), ' data points of class ', cluster_labels[i])
        print( '-' * 50 )

        # Calculating the purity of the clusters and printing the report
        calculate_purity( clusters, labels, print_report=True )

        # Printing the count for each unique labels in each cluster horizontally
        for i in range(len(cluster_indices)):
            print( bold('[Cluster #' + str(i+1) + ']'), ' --> ', textf(cluster_labels[i], COLOR_CYAN) )
            print(labels[cluster_indices[i]].value_counts())
            print('-' * 50)
    try:
        centroids, clusters, cluster_indices, cluster_labels= kmeans_clustering( k=nbc, data=X_train, print_updates=True)
    except KeyboardInterrupt:
        print( textf('Process interrupted by user', COLOR_RED) )
#process('corrected','kddcup.data_10_percent_corrected',5)
   
