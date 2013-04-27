# Import the Required Libraries
import numpy as np
from StringIO import StringIO
import math 
import random

# Minimzes the l2 error over user vectors keeping movie vector constant
def user_optimization(rating_matrix, user_vectors, movie_vectors, K):
    # Set up the constants for use WHAT VALUES TO USE
    # alpha is learning rate and lambda_r is regularization parameter
    MAX_ITERATIONS = 5000
    alpha = 0.0002
    lambda_r = 0.02
    
    # Take the transpose of the item_vector array
    movie_vectors = movie_vectors.T

    for step in xrange(MAX_ITERATIONS):
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                # Only if User has rated the movie
                if rating_matrix[i][j] > 0:
                	# Calculate the Error
                    error = rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j])

                    # Change the Vectors depending on the derivative, only change movie_vector: ALTERNATING OPTIMIZATION
                    for k in xrange(K):
                        user_vectors[i][k] = user_vectors[i][k] + alpha * (2 * error * movie_vectors[k][j] - lambda_r * user_vectors[i][k])
                        # movie_vectors[k][j] = movie_vectors[k][j] + alpha * (2 * error * user_vectors[i][k] - lambda_r * movie_vectors[k][j])
        
        # Calculate the Total Error
        error = 0
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    error = error + pow(rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j]), 2)
                    for k in xrange(K):
                        error = error + (lambda_r) * (pow(user_vectors[i][k],2))

        # Break if converged        
        if error < 0.001:
            break

    # return the movie profile vectors
    return user_vectors, movie_vectors.T


# Minimzes the l2 error over movie vectors keeping user vectors constant
def movie_optimization(rating_matrix, user_vectors, movie_vectors, K):
    # Set up the constants for use WHAT VALUES TO USE
    # alpha is learning rate and lambda_r is regularization parameter
    MAX_ITERATIONS = 5000
    alpha = 0.0002
    lambda_r = 0.02
    
    # Take the transpose of the item_vector array
    movie_vectors = movie_vectors.T

    for step in xrange(MAX_ITERATIONS):
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                # Only if User has rated the movie
                if rating_matrix[i][j] > 0:
                	# Calculate the Error
                    error = rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j])

                    # Change the Vectors depending on the derivative, only change movie_vector: ALTERNATING OPTIMIZATION
                    for k in xrange(K):
                        # user_vectors[i][k] = user_vectors[i][k] + alpha * (2 * error * movie_vectors[k][j] - lambda_r * user_vectors[i][k])
                        movie_vectors[k][j] = movie_vectors[k][j] + alpha * (2 * error * user_vectors[i][k] - lambda_r * movie_vectors[k][j])
        
        # Calculate the Total Error
        error = 0
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    error = error + pow(rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j]), 2)
                    for k in xrange(K):
                        error = error + (lambda_r) * (pow(movie_vectors[k][j],2))

        # Break if converged
        if error < 0.001:
            break

    # return the movie profile vectors
    return movie_vectors

# Minimzes the l2 error over user vectors keeping movie vector constant
def user_optimization_heirarchical(rating_matrix, user_vectors, movie_vectors, parent_user_vector, K):
    # Set up the constants for use WHAT VALUES TO USE
    # alpha is learning rate and lambda_r is regularization parameter
    MAX_ITERATIONS = 5000
    alpha = 0.0002
    lambda_r = 0.02
    
    # Take the transpose of the item_vector array
    movie_vectors = movie_vectors.T

    for step in xrange(MAX_ITERATIONS):
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                # Only if User has rated the movie
                if rating_matrix[i][j] > 0:
                    # Calculate the Error
                    error = rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j])

                    # Change the Vectors depending on the derivative, only change movie_vector: ALTERNATING OPTIMIZATION
                    for k in xrange(K):
                        user_vectors[i][k] = user_vectors[i][k] + alpha * (2 * error * movie_vectors[k][j] - lambda_r * (user_vectors[i][k] - parent_user_vector[i][k]))
                        # movie_vectors[k][j] = movie_vectors[k][j] + alpha * (2 * error * user_vectors[i][k] - lambda_r * movie_vectors[k][j])
        
        # Calculate the Total Error
        error = 0
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    error = error + pow(rating_matrix[i][j] - np.dot(user_vectors[i,:],movie_vectors[:,j]), 2)
                    for k in xrange(K):
                        error = error + (lambda_r) * (pow(user_vectors[i][k] - parent_user_vector[i][k],2))

        # Break if converged        
        if error < 0.001:
            break

    # return the movie profile vectors
    return user_vectors, movie_vectors.T