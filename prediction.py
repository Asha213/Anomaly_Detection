def process(trainpath,testpath,noc):
    import matplotlib.pyplot as plt
    import pandas as pd
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
    import csv
    import Randomfo as rfmodel
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
   
    # try:
    #     centroids15, clusters15, cluster_indices15, cluster_labels15 = kmeans_clustering( k=15, data=X_train, print_updates=True, save_cp=False )
    # except KeyboardInterrupt:
    #     print( textf('Process interrupted by user', COLOR_RED) )
    # centroids23, clusters23, cluster_indices23, cluster_labels23 = load_state_kmeans(23)
    # try:
    #     centroids23, clusters23, cluster_indices23, cluster_labels23 = kmeans_clustering( k=23, data=X_train, print_updates=True, initial_centroids=centroids23 )
    # except KeyboardInterrupt:
    #     print( textf('Process interrupted by user', COLOR_RED) )
    # try:
    #     centroids31, clusters31, cluster_indices31, cluster_labels31 = kmeans_clustering( k=31, data=X_train, print_updates=True )
    # except KeyboardInterrupt:
    #     print( textf('Process interrupted by user', COLOR_RED) )
    # try:
    #     centroids45, clusters45, cluster_indices45, cluster_labels45 = kmeans_clustering( k=45, data=X_train, print_updates=True )
    # except KeyboardInterrupt:
    #     print( textf('Process interrupted by user', COLOR_RED) )
    def test_kmeans( k, test_data, centroids, cluster_labels ):
        
        y_pred, y_actual = [], []

        for i in range( test_data.shape[0] ):

            if y_test.iloc[i] not in cluster_labels.values(): continue

            # Calculating the distance of the data point from each centroid
            distances = [ np.linalg.norm( test_data.iloc[i] - centroids.iloc[j] ) for j in range(k) ]

            # Finding the index of the centroid with the minimum distance
            min_index = np.argmin(distances)

            # Assigning the label of the centroid to the data point
            y_pred.append( cluster_labels[min_index] )
            y_actual.append( y_test.iloc[i] )

        return y_actual, y_pred
        # This function will be used to evaluate the model using sklearn's functions
    def evaluate_model( y_test, y_pred, clusters: dict, avg ):
            precision = precision_score( y_test, y_pred, average=avg )
            recall = recall_score( y_test, y_pred, average=avg )
            f1 = f1_score( y_test, y_pred, average=avg )
            accuracy = accuracy_score( y_test, y_pred )

            labels, counts = np.unique(y_test, return_counts=True)
            conditional_entropy = -np.sum( counts / np.sum(counts) * np.log2( counts / np.sum(counts) ) )
            conditional_entropy = conditional_entropy / len(clusters)
        
            return precision, recall, f1, accuracy, conditional_entropy
    centroids7, clusters7, cluster_indices7, cluster_labels7 = load_state_kmeans(noc)
    y7, y_pred7 = test_kmeans( k=noc, test_data=X_test, centroids=centroids7, cluster_labels=cluster_labels7 )
    print( len(y7), len(y_pred7) )
    precision7_macro, recall7_macro, f1_7_macro, accuracy7, cond_entropy7 = evaluate_model( y7, y_pred7, clusters7, avg='macro' )
    precision7_weighted, recall7_weighted, f1_7_weighted, accuracy7, cond_entropy7= evaluate_model( y7, y_pred7, clusters7, avg='weighted' )
    print( 'K = :',noc )
    print( '------' )

    print( 'Precision (macro): ', precision7_macro )
    print( 'Recall (macro): ', recall7_macro )
    print( 'F1 Score (macro): ', f1_7_macro )
    print( '-' * 50 )

    print( 'Precision (weighted): ', precision7_weighted )
    print( 'Recall (weighted): ', recall7_weighted )
    print( 'F1 Score (weighted): ', f1_7_weighted )
    print( '-' * 50 )

    print( 'Accuracy: ', accuracy7 )
    print( 'Conditional Entropy: ', cond_entropy7 )
    print( '-' * 50 )
    result2=open('results/K_meansMetrics.csv', 'w')
    result2.write("Parameter,Value" + "\n")
    result2.write("Precision" + "," +str(precision7_weighted) + "\n")
    result2.write("Recall" + "," +str(recall7_weighted) + "\n")
    result2.write("F1 Score" + "," +str(f1_7_weighted) + "\n")
    result2.write("Conditional Entropy" + "," +str(cond_entropy7) + "\n")
    result2.write("ACCURACY" + "," +str(accuracy7) + "\n")
    result2.close()
    df =  pd.read_csv('results/K_meansMetrics.csv')
    acc = df["Value"]
    alc = df["Parameter"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    explode = (0.1, 0, 0, 0, 0)  
    fig = plt.figure()
    plt.bar(alc, acc,color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('K-Means Metrics Value for'+str(noc))
    fig.savefig('results/K_Means_MetricsValue.png') 
    plt.pause(5)
    plt.show(block=False)
    plt.close()
        

    print( classification_report( y7, y_pred7 ) )
    rfmodel.process(trainpath,testpath)
   
#process('corrected','kddcup.data_10_percent_corrected',5)
