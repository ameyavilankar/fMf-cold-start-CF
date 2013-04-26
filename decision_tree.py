# Import the Required Libraries
import numpy as np
from StringIO import StringIO
import math 

# import form user defined libraries
import optimization as opt

# This class represents each Node of the Decision Tree
# TODO: Add more data members
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
		
		# TODO: ADD user profile vector assoicated with each node

class Tree:
	# __init__() sets the root node, currentDepth and maxdepth of the tree
	def __init__(self, root_node, leaf_size = 1, max_depth = 7):
		self.root = root_node
		self.max_depth = max_depth

	# recursively builds up the entire tree from the root Node	
	def fitTree(self, current_node, rating_matrix, movie_vectors, K):
		# rating_matrix only consists of rows which are users corresponding to the current Node

		# Check if the maxDepth is reached
		if current_node.depth > max_depth:
			return

		# Update the depth
		currentDepth = currentDepth + 1
		
		# Set the depth of the current Node
		current_node.depth = currentDepth

		#Create a numy_array to hold the split_criteria Values
		split_values = np.zeros(rating_matrix[0])

		for movie_index in xrange(len(rating_matrix[0])):
			# Split the rating_matrix into like, dislike and unknown based on the current MovieIndex
			(like, dislike, unknown) = splitUsers(rating_matrix, movie_index)

			#Generate a random vector as seed for the process
			# Get the like, dislike and unknown user vector
			# TODO: Normal Regularization...USE Hierarchical Regularization
			random_vectors = np.random.rand(len(like), K)
			like_vector = opt.user_optimization(like, random_vectors, movie_vectors, K)

			random_vectors = np.random.rand(len(dislike), K)
			dislike_vector = opt.user_optimization(dislike, random_vectors, movie_vectors, K)
			
			random_vectors = np.random.rand(len(unknown), K)
			unknown_vector = opt.user_optimization(unknown, random_vectors, movie_vectors, K)

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
		(like, dislike, unknown) = splitUsers(rating_matrix, bestMovie)

		# TODO: Set the user profile vector for the current Node..after addding it to the node class

		# TODO CONDITION check condition RMSE Error check
		if True:
			# Recursively call the fitTree function for like, dislike and unknown Nodes creation
			current_node.like = Node(current_node, current_node.depth + 1)
			self.fitTree(current_node.like, like, movie_vectors, K)

			current_node.dislike = Node(current_node, current_node.depth + 1)
			self.fitTree(current_node.dislike, dislike, movie_vectors, K)
			
			current_node.unknown = Node(current_node, current_node.depth + 1)
			self.fitTree(current_node.unknown, unknown, movie_vectors, K)

		return


