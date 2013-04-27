# Import the Required Libraries
import numpy as np
from StringIO import StringIO
import math 

# import form user defined libraries
import optimization as opt

# This class represents each Node of the Decision Tree
class Node:
	def __init__(self, parent_node, node_depth):
		# Each Node has a Like, Dislike and Unknown Child
		# It also stores the index of the movie on which its splits the data
		self.parent = parent_node
		self.depth = node_depth
		self.like = None
		self.disklike = None
		self.unknown = None
		self.movie_index = None
		self.user_vector = None

class Tree:
	# __init__() sets the root node, currentDepth and maxdepth of the tree
	def __init__(self, root_node, max_depth = 7):
		self.root = root_node
		self.max_depth = max_depth
		self.ultimate_user_vector = None

	# recursively builds up the entire tree from the root Node	
	def fitTree(self, current_node, rating_matrix, movie_vectors, K):
		# rating_matrix only consists of rows which are users corresponding to the current Node

		# Check if the maxDepth is reached
		if current_node.depth > max_depth:
			return

		# Calulate the Error Before the Split
		error_before = 0
        for i in xrange(len(rating_matrix)):
            for j in xrange(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    error_before = error_before + pow(rating_matrix[i][j] - np.dot(user_vectors[i,:].T, movie_vectors[:,j]), 2)
                    # TODO: Check if regularized
                    # for k in xrange(K):
                    #    error_before = error_before + (lambda_r) * (pow(user_vectors[i][k],2))

		#Create a numy_array to hold the split_criteria Values
		split_values = np.zeros(rating_matrix[0])

		for movie_index in xrange(len(rating_matrix[0])):
			# Split the rating_matrix into like, dislike and unknown based on the current MovieIndex
			(like, dislike, unknown) = splitUsers(rating_matrix, movie_index)

			#Generate a random vector as seed for the process
			# Get the like, dislike and unknown user vector
			# TODO: Normal Regularization...USE Hierarchical Regularization
					# Split the rating_matrix into like, dislike and unknown
			(indices_like, indices_dislike, indices_unknown) = splitUsers(rating_matrix, bestMovie)

			like = rating_matrix[indices_like]
			parent_like_vector = current_node.user_vector[indices_like]

			dislike = rating_matrix[indices_dislike]
			parent_dislike_vector = current_node.user_vector[indices_dislike]

			unknown = rating_matrix[indices_unknown]
			parent_unknown_vector = current_node.user_vector[indices_unknown]

			# Calculate the User Profile Vector for each of the three classes
			random_vectors = np.random.rand(len(like), K)
			like_vector = opt.user_optimization_heirarchical(like, random_vectors, movie_vectors, parent_like_vector, K)

			random_vectors = np.random.rand(len(dislike), K)
			dislike_vector = opt.user_optimization_heirarchical(dislike, random_vectors, movie_vectors, parent_dislike_vector, K)
			
			random_vectors = np.random.rand(len(unknown), K)
			unknown_vector = opt.user_optimization_heirarchical(unknown, random_vectors, movie_vectors, parent_unknown_vector, K)
			
			# Calculate the split criteria value
			value = 0

			# Add the like part
			for i in xrange(len(like)):
				for j in xrange(len(like[i])):
					if like[i][j] > 0:
						value = value + pow(like[i][j] - np.dot(like_vector[i, :], movie_vectors[:,j].T), 2)

            # Add the dislike part
			for i in xrange(len(dislike)):
				for j in xrange(len(dislike[i])):
					if dislike[i][j] > 0:
						value = value + pow(dislike[i][j] - np.dot(dislike_vector[i, :], movie_vectors[:,j].T), 2)

            # Add the unknown part
			for i in xrange(len(unknown)):
				for j in xrange(len(unknown[i])):
					if unknown[i][j] > 0:
						value = value + pow(unknown[i][j] - np.dot(unknown_vector[i, :], movie_vectors[:,j].T), 2)
			
			# Store the split criteria values for current movie_index
			split_values[movie_index]  = value

		# Get the index of the movie with the maximum split value
		bestMovie = np.argmax(split_values)

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
		random_vectors = np.random.rand(len(like), K)
		like_vector = opt.user_optimization_heirarchical(like, random_vectors, movie_vectors, parent_like_vector, K)

		random_vectors = np.random.rand(len(dislike), K)
		dislike_vector = opt.user_optimization_heirarchical(dislike, random_vectors, movie_vectors, parent_dislike_vector, K)
		
		random_vectors = np.random.rand(len(unknown), K)
		unknown_vector = opt.user_optimization_heirarchical(unknown, random_vectors, movie_vectors, parent_unknown_vector, K)		

		# CONDITION check condition RMSE Error check is CORRECT
		if split_values[bestMovie] < error_before:
			# Recursively call the fitTree function for like, dislike and unknown Nodes creation
			current_node.like = Node(current_node, current_node.depth + 1)
			current_node.like.user_vector = like_vector
			self.fitTree(current_node.like, like, movie_vectors, K)

			current_node.dislike = Node(current_node, current_node.depth + 1)
			current_node.dislike.user_vector = dislike_vector
			self.fitTree(current_node.dislike, dislike, movie_vectors, K)
			
			current_node.unknown = Node(current_node, current_node.depth + 1)
			current_node.unknown.user_vector = unknown_vector
			self.fitTree(current_node.unknown, unknown, movie_vectors, K)

		return


	# fucntion used to traverse a tree based on the answers
	def traverse(user_answers):

		current_node = root

		while current_node != None: 
			if user_answers[current_node.movie_index] == 0:
				current_node = root.like
			elif user_answers[current_node.movie_index] == 1:
				current_node = root.dislike
			else:
				current_node = root.unknown_vector

	# Returns the user vector for the decision tree
	def getUserVectors():
		getLeafVector(self.root)
		return np.asarray(self.ultimate_user_vector)

	# Recursively builds up the user profile vector matrix by adding profiles at the leaves
	def getLeafVector(current_node):
		# If Leaf node, append user vector to ulitmate user vector
		if current_node.like == None and current_node.dislike == None and current_node.unknown == None:
			self.ultimate_user_vector.append(current_node.user_vector)

		# if not lead node, call recursively on like, dislike and unknown
		getLeafVector(current_node.like)
		getLeafVector(current_node.dislike)
		getLeafVector(current_node.unknown)

