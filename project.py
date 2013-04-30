# Import the Required Libraries
import numpy as np
from StringIO import StringIO
import math 
import random

# import form user defined libraries
import decision_tree as dtree
import optimization as opt

# get the data from the MovieLens Dataset
def getRatingMatrix(filename):
    # Open the file for reading data
    file = open(filename, "r")

    while 1:
        # Read all the lines from the file and store it in lines
        lines = file.readlines(1000000000)

        # if Lines is empty, simply break
        if not lines:
            break

        # Create a Dictionary of DIctionaries
        User_Movie_Dict = {}

        # Create a list to hold all the data
        data = []

        print "Number of Lines: ", len(lines)

        # For each Data Entry, get the Y and the Xs in their respective list
        for line in lines:
            # Get all the attributes by splitting on '::' and 
            # use list comprehension to convert string to float
            list1 = line.split("\n")[0].split("::")
            list1 = [float(j) for j in list1]

            # Add to the data
            data.append(list1)

        # Convert the data into numpyarray        
        data_array = np.array(data)

        # Get the indices of the maximum Values in each column
        a = np.argmax(data, axis=0)
        num_users = data[a[0]][0]
        num_movies = data[a[1]][1]

        # print "Max values Indices: ", a
        print "Number of Users: ", num_users
        print "Number of Movies: ", num_movies

        # Creat and initialise Rating Matrix to hold all the rating values
        ratingMatrix = np.zeros((num_users, num_movies))

        for list1 in data:
            # print list1[0], " ", list1[1]
            ratingMatrix[list1[0] - 1][list1[1] - 1] = list1[2]
            
        # Return both the array and the dict
        return (User_Movie_Dict, ratingMatrix)


def cf_movie_vector(rating_matrix, user_vectors, movie_vectors, K):
    lambda_r = 0.02
    
    # Stores the user profile vectors
    movie_profiles = np.zeros((len(rating_matrix[0]), K))
    
    count = 0
    for j in xrange(len(rating_matrix[0])):
        first_term = np.zeros((K, K))
        
        for i in xrange(len(rating_matrix)):
            if rating_matrix[i][j] > 0:     
                first_term = np.add(first_term, np.outer(user_vectors[i], user_vectors[i]))

        # Take the inverse of the first term
        first_term = np.add(first_term, np.multiply(lambda_r, np.eye(K)))
        first_term = np.linalg.inv(first_term)

        second_term = np.zeros(K)

        for i in xrange(len(user_vectors)):
            if rating_matrix[i][j] > 0:
                second_term = np.add(second_term, np.multiply(rating_matrix[i][j], user_vectors[i]))

        movie_profiles[count] = np.dot(first_term, second_term)
        count = count + 1

    return movie_profiles


# Function to calculate the RMSE Error between the predicted and actual rating
def getRMSE(Actual_Rating, Predicted_Rating):
    # Calculate the Root Mean Squared Error(RMS)
    rmse = 0.0
    for i in xrange(len(Actual_Rating)):
        for j in xrange(len(Actual_Rating[0])):
            if Actual_Rating[i][j] > 0:
                rmse = rmse + pow((Actual_Rating[i][j] - Predicted_Rating[i][j]), 2)

    rms = rmse * 1.0 / len(Actual_Rating)
    rmse = math.sqrt(rmse)

    # Print and return the RMSE
    print 'Root Mean Squared Error(RMS) = ' , rms
    return rms


# Used to randomly split the data
def random_split(data):
    # Split the data set into 75% and 25%
    SPLIT_PERCENT = 0.75
    
    # Get Random Indices to shuffle the rows around
    indices = np.random.permutation(data.shape[0])

    # Get the number of rows 
    num_rows = len(data[:, 0])

    # Get the indices for training and testing sets
    training_indices, test_indices = indices[: int(SPLIT_PERCENT * num_rows)], indices[int(SPLIT_PERCENT * num_rows) :]

    # return the training and the test set
    return data[training_indices,:], data[test_indices,:]


# Returns the rating Matrix with approximated ratings for all users for all movies using fMf
def alternateOptimization(rating_matrix, NUM_OF_FACTORS):
    # Save and print the Number of Users and Movies
    NUM_USERS = rating_matrix.shape[0]
    NUM_MOVIES = rating_matrix.shape[1]
    print "Number of Users", NUM_USERS
    print "Number of Movies", NUM_MOVIES
    print "Number of Latent Factors: ", NUM_OF_FACTORS

    # Create the user and item profile vector of appropriate size.
    # Initialize the item vectors randomly, check the random generation
    user_vectors = np.zeros((NUM_USERS, NUM_OF_FACTORS), dtype=float)
    movie_vectors = np.random.rand(NUM_MOVIES, NUM_OF_FACTORS)

    i = 0    
    
    print "Entering Main Loop of alternateOptimization"

    decTree = dtree.Tree(dtree.Node(None, 1), rating_matrix, NUM_OF_FACTORS)

    # Do converge Check
    while i < 10:
        # Create the decision Tree based on movie_vectors
        #print "Creating tree..."

        decTree = dtree.Tree(dtree.Node(None, 1), rating_matrix, NUM_OF_FACTORS)
        
        #print "Creating Tree.. for i = ", i

        decTree.fitTree(decTree.root, rating_matrix, movie_vectors, NUM_OF_FACTORS)

        #print "Tree Created for i = ", i
        #print "Getting the user vectors from tree"

        # Calculate the User vectors using dtree
        user_vectors = decTree.getUserVectors(rating_matrix, NUM_OF_FACTORS)

        #print "Got the user vectors from the decisionTree"
        #print "Optimizing movie_vectors.."

        # Optimize Movie vector using the calculated user vectors
        movie_vectors = cf_movie_vector(rating_matrix, user_vectors, movie_vectors, NUM_OF_FACTORS)

        #print "movie_vectors Optimized"
        
        # Calculate Error for Convergence check
        i = i + 1

    # return the completed rating matrix    
    return (decTree, movie_vectors.T)


def printTopKMovies(test, predicted, K = 2):
    # Gives top K (2) recommendations
    K = 2

    zero_list = []
    movie_list = []

    for i in xrange(len(test)):
        for j in xrange(len(test[0])):
            if test[i][j] == 0:
                zero_list.append(predicted[i][j])
                movie_list.append(j)

            zero_array = np.array(zero_list)
            movie_array = np.array(movie_list)

            args = np.argsort(zero_array)
            movie_array = movie_array[args]

            print "Top Movies Not rated by the user"
            print movie_array[0:K-1]

if __name__ == "__main__":
    # Get the Data
    (User_Movie_Dict, data) = getRatingMatrix("ratings_small.dat")

    print "Dimensions of the Dataset: ", data.shape
    
    # Split the data 75-25 into training and testing dataset
    (train, test) = random_split(data)
    print "Dimensions of the Training Set: ", train.shape
    print "Dimensions of the Testing Set: ", test.shape

    # Split the testing dataset 75-25 into answer and evaluation dataset
    (answer, evaluation) = random_split(test)
    print "Dimensions of the Answer Set: ", answer.shape
    print "Dimensions of the Evaluation Set: ", evaluation.shape
    
    # Set the number of Factors
    NUM_OF_FACTORS = 20
    
    (decisionTree, movie_vector) = alternateOptimization(train, NUM_OF_FACTORS = 20)
    user_vectors = decisionTree.getUserVectors(test, NUM_OF_FACTORS)

    Predicted_Rating = np.dot(user_vectors, movie_vector)
    print "Predicted_Rating for Test: ", Predicted_Rating
    print "Test Rating: ", test
    print "RMSE on Testing: ", getRMSE(test, Predicted_Rating)

    # Top K new recommendations:
    # printTopKMovies(test, Predicted_Rating, 1)

    # Testing
    # testMatrix = np.array([[0, 5, 3, 4, 0], [3, 4, 0, 3, 1], [3, 0, 4, 0, 2], [4, 4, 4, 3, 0], [3, 5, 0, 4, 0]])
    # (indices_like, indices_dislike, indices_unknown) = splitUsers(data, 2293)
    
    # like = data[indices_like]
    # dislike = data[indices_dislike]
    # unknown = data[indices_unknown]

    # print "Dimensions of the Like: ", like.shape[0], like.shape[1]
    # print "Dimensions of the Dislike: ", dislike.shape[0], dislike.shape[1]
    # print "Dimensions of the Unknown: ", unknown.shape[0], unknown.shape[1]
    
    # Get the decision tree and the item profile vectors using alternate optimization
    #(decisionTree, item_vector) = alternateOptimization(train)
    # train = test = np.array([[0, 5, 3, 4, 0], [3, 4, 0, 3, 1], [3, 0, 4, 0, 2], [4, 4, 4, 3, 0], [3, 5, 0, 4, 0]])
    