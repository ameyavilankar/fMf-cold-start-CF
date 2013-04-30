# Import the Required Libraries
import numpy as np
from StringIO import StringIO
import math 

# import form user defined libraries
import optimization as opt


# Split the users into Like, Dislike and Unknown Users
def splitUsers(data, movie_index):
    # Get the indices for the when the rating is greater than 3: LIKE
    indices_like = np.where(data[:, movie_index] > 3.0)[0]

    # Get the indices for the when the rating is less than 3: DISLIKE
    indices_dislike = np.where((data[:, movie_index] <= 3.0) & (data[:, movie_index] != 0))[0]

    # Get the indices for the when the rating is equal to 0: UNKNOWN
    indices_unknown = np.where(data[:, movie_index] == 0)[0]
    
    # print data[indices_like, :][:, movie_index]
    # print data[indices_dislike, :][:, movie_index]
    # print data[indices_unknown, :][:, movie_index]

    return indices_like, indices_dislike, indices_unknown
    # return data[indices_like, :], data[indices_dislike, :], data[indices_unknown, :]


def closed_form(rating_matrix, movie_vectors, indices, K):
	# Stores the user profile vectors
	user_profiles = np.zeros((len(indices), K))
	index_matrix = rating_matrix[indices]
	
	count = 0
	for i in indices:
		first_term = np.zeros((K, K))
		
		for j in xrange(len(movie_vectors)):
			first_term = np.add(first_term, np.outer(movie_vectors[j], movie_vectors[j]))

		# Take the inverse of the first term
		first_term = np.linalg.inv(first_term)

		second_term = np.zeros(K)

		for j in xrange(len(movie_vectors)):
			second_term = np.add(second_term, np.multiply(rating_matrix[i][j], movie_vectors[j]))

		user_profiles[count] = np.dot(first_term, second_term)
		count = count + 1

	return user_profiles

# This class represents each Node of the Decision Tree
class Node:
	def __init__(self, parent_node, node_depth):
		# Each Node has a Like, Dislike and Unknown Child
		# It also stores the index of the movie on which its splits the data
		self.parent = parent_node
		self.depth = node_depth
		self.like = None
		self.dislike = None
		self.unknown = None
		self.movie_index = None
		self.user_vector = None

class Tree:
	# __init__() sets the root node, currentDepth and maxdepth of the tree
	def __init__(self, root_node, rating_matrix, K, max_depth = 7):
		self.root = root_node
		self.root.user_vector = np.random.rand(len(rating_matrix), K)
		self.max_depth = max_depth

	# fucntion used to traverse a tree based on the answers
	def traverse(self, user_answers):
		# rand_prob[rand_prob > prob_failure] = 1
		current_node = self.root

		print "Before"
		# Traverse the tree till you reach the leaf
		while current_node.like != None or current_node.dislike != None or current_node.unknown != None : 
			if user_answers[current_node.movie_index] == 0:
				current_node = current_node.like
			elif user_answers[current_node.movie_index] == 1:
				current_node = current_node.dislike
			else:
				current_node = current_node.unknown
		
		# return the user vecotr associated with the lead node
		print "zzz", current_node.user_vector.shape
		return np.mean(current_node.user_vector, axis = 0)

	# Returns the user vector for the decision tree
	def getUserVectors(self, rating_matrix, K):
		ultimate_user_vector = np.zeros((len(rating_matrix), K))
		
		for i in xrange(len(rating_matrix)):
			# Stores the user response
			user_response = np.zeros(len(rating_matrix[0]))
			
			# Get the responses using the rating matrix
			for j in xrange(len(rating_matrix[0])):
				if rating_matrix[i][j] > 3:
					user_response[j] = 0
				elif rating_matrix[i][j] == 0:
					user_response[j] = 2
				else:
					user_response[j] = 1

			# Traverse the tree abd store the user vector associated with leaf node reached
			temp = self.traverse(user_response)
			print "zz", temp.shape
			ultimate_user_vector[i] = temp

		# return the user vector
		return ultimate_user_vector

	# recursively builds up the entire tree from the root Node	
	def fitTree(self, current_node, rating_matrix, movie_vectors, K):
		# rating_matrix only consists of rows which are users corresponding to the current Node
		# Check if the maxDepth is reached
		if current_node.depth > self.max_depth:
			return

		#print "rating_matrix: ", rating_matrix.shape
		#print "User matrix: ", current_node.user_vector.shape
		#print "movie_vectors: ", movie_vectors.shape

		# Calulate the Error Before the Split
		error_before = 0
		for i in xrange(len(rating_matrix)):
			for j in xrange(len(rating_matrix[i])):
				if rating_matrix[i][j] > 0:
					error_before = error_before + pow(rating_matrix[i][j] - np.dot(current_node.user_vector[i,:].T, movie_vectors[j, :]), 2)
	                # TODO: Check if regularized
	                # for k in xrange(K):
	                #    error_before = error_before + (lambda_r) * (pow(user_vectors[i][k],2))

	    #Create a numy_array to hold the split_criteria Values
		split_values = np.zeros(len(rating_matrix[0]))

		# print "Error Before: ", error_before

		for movie_index in xrange(len(rating_matrix[0])):
			# Split the rating_matrix into like, dislike and unknown
			(indices_like, indices_dislike, indices_unknown) = splitUsers(rating_matrix, movie_index)

			like = rating_matrix[indices_like]
			parent_like_vector = current_node.user_vector[indices_like]

			dislike = rating_matrix[indices_dislike]
			parent_dislike_vector = current_node.user_vector[indices_dislike]

			unknown = rating_matrix[indices_unknown]
			parent_unknown_vector = current_node.user_vector[indices_unknown]

			# print "Split the data into like, disklike and unknown for movie", movie_index

			# Calculate the User Profile Vector for each of the three classes
			like_vector = closed_form(rating_matrix, movie_vectors, indices_like, K)
			dislike_vector = closed_form(rating_matrix, movie_vectors, indices_dislike, K)
			unknown_vector = closed_form(rating_matrix, movie_vectors, indices_unknown, K)

			# print "Like vector: ", like_vector.shape
			# print "Disike vector: ", dislike_vector.shape
			# print "Unknown vector: ", unknown_vector.shape

			# print "Like matrix", like.shape
			# print "DisLike matrix", dislike.shape
			# print "Unknown matrix", unknown.shape
			
			# Calculate the split criteria value
			value = 0

			# Add the like part
			for i in xrange(len(like)):
				for j in xrange(len(like[i])):
					if like[i][j] > 0:
						# print "1: ", like_vector[i, :].shape, "2: ", movie_vectors[j,:].shape
						value = value + pow(like[i][j] - np.dot(like_vector[i, :], movie_vectors[j, :]), 2)

	        # Add the dislike part
			for i in xrange(len(dislike)):
				for j in xrange(len(dislike[i])):
					if dislike[i][j] > 0:
						value = value + pow(dislike[i][j] - np.dot(dislike_vector[i, :], movie_vectors[j, :]), 2)

	        # Add the unknown part
			for i in xrange(len(unknown)):
				for j in xrange(len(unknown[i])):
					if unknown[i][j] > 0:
						value = value + pow(unknown[i][j] - np.dot(unknown_vector[i, :], movie_vectors[j, :]), 2)
			
			# Store the split criteria values for current movie_index
			split_values[movie_index]  = value

		# Get the index of the movie with the maximum split value
		bestMovie = np.argmax(split_values)

		#print "bestMovie index: ", bestMovie

		# Store the movie_index for the current_node
		current_node.movie_index = bestMovie

		# Split the rating_matrix into like, dislike and unknown
		(indices_like, indices_dislike, indices_unknown) = splitUsers(rating_matrix, bestMovie)

		like = rating_matrix[indices_like]
		parent_like_vector = current_node.user_vector[indices_like]

		dislike = rating_matrix[indices_dislike]
		parent_dislike_vector = current_node.user_vector[indices_dislike]

		unknown = rating_matrix[indices_unknown]
		parent_unknown_vector = current_node.user_vector[indices_unknown]

		# Calculate the User Profile Vector for each of the three classes
		# print "optimizing like, dislike and unknown..."
	
		# Calculate the User Profile Vector for each of the three classes
		like_vector = closed_form(rating_matrix, movie_vectors, indices_like, K)
		dislike_vector = closed_form(rating_matrix, movie_vectors, indices_dislike, K)
		unknown_vector = closed_form(rating_matrix, movie_vectors, indices_unknown, K)



		# CONDITION check condition RMSE Error check is CORRECT
		if split_values[bestMovie] < error_before:
			# Recursively call the fitTree function for like, dislike and unknown Nodes creation
			current_node.like = Node(current_node, current_node.depth + 1)
			current_node.like.user_vector = like_vector
			if len(like) != 0:
				self.fitTree(current_node.like, like, movie_vectors, K)

			current_node.dislike = Node(current_node, current_node.depth + 1)
			current_node.dislike.user_vector = dislike_vector
			if len(dislike) != 0:
				self.fitTree(current_node.dislike, dislike, movie_vectors, K)
			
			current_node.unknown = Node(current_node, current_node.depth + 1)
			current_node.unknown.user_vector = unknown_vector
			if len(unknown) != 0:
				self.fitTree(current_node.unknown, unknown, movie_vectors, K)