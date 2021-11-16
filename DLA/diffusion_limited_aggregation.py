import numpy as np
from tqdm import tqdm
from time import time
from random import choice, random

class DiffusionLimitedAggregation(object):

	def __init__(self, shape=(251,251)):

		# Assert constraints on inputs
		assert len(shape)==2, "Invalid shape parameter. Only 2D matrices supported."
		assert (shape[0]%2==1) and (shape[1]%2==1), "Both dimensions should have odd number of pixels"

		# Define parameters for DLA
		self.shape = shape

		# Set initial image with only center as 1
		self.image = np.zeros(shape, dtype=int)
		self.image[shape[0]//2, shape[1]//2] = 1

		# Store indices for potential starting positions
		self.starting_idxs = self._generate_starting_idxs()


	def _generate_starting_idxs(self):
		'''
			Method to generate a list of coordinates corresponding
			to points on the corner of the square. New particles can
			only be added into the system through these points.
		'''

		# Estimate all possible row and column values
		nrow, ncol = self.shape
		row_idxs = range(nrow)
		col_idxs = range(ncol)

		# List down coordinates for top and bottom rows
		top_row = list(zip([0]*ncol, col_idxs))
		bottom_row = list(zip([nrow-1]*ncol, col_idxs))

		# List down coordinates for left and right columns
		left_col = list(zip(row_idxs[1:-1], [0]*nrow))
		right_col = list(zip(row_idxs[1:-1], [ncol-1]*nrow))

		return top_row + bottom_row + left_col + right_col


	def _add_particle(self):
		'''
			Method to add a new particle into the system.
			The particle can randomly enter through any of the
			starting indices.
		'''
		# Randomly select a point from the list of starting indices
		row, col = choice(self.starting_idxs)
		self.image[row, col] = 1
		return row, col


	@staticmethod
	def _is_stuck(stickiness):
		'''
			Method to check if the particle gets stuck to an
			existnig particle, based on the stickiness value
			of the particles.
		'''
		if random() < stickiness:
			output = True
		else:
			output = False
		return output


	def _run_random_walk(self, start, stickiness):
		
		'''
			Method to run random walks given a strating point and
			a stickiness value. The random walk ends as soon as the
			new particles gets stuck to an existing particle.
		'''

		# Assert constraints on inputs
		assert self.image[start[0], start[1]] == 1

		# Define limits for random walks
		min_row = min_col = 0
		max_row, max_col = self.shape

		# Set variables to track random walk
		walk_status = True
		
		# Run random walk
		while walk_status:

			# Get neighboring pixels
			row, col = start[0], start[1]
			rmin, rmax = max(row-1, min_row), min(row+1, max_row)
			cmin, cmax = max(col-1, min_col), min(col+1, max_col)
			neighbors = self.image[rmin:rmax+1, cmin:cmax+1]

			# Check if a non-zero neighbor exists and
			# if the current point gets stuck to the
			# non-zero neighbor
			idxs = list(neighbors.nonzero())
			idxs[0] += max(row-1, min_row)
			idxs[1] += max(col-1, min_col)
			idxs = list(zip(*idxs))
			idxs.remove((row,col))
			
			# Check if the point gets stuck to any non-zero neighbor
			sticky_status = [self._is_stuck(stickiness) for _ in idxs]
			if any(sticky_status) or len(idxs)==neighbors.size-1:
				walk_status = False
				if (row in (min_row, max_row)) or (col in (min_col, max_col)):
					self.starting_idxs.remove((row,col))

			else:
				# Get indices of all non-zero neighbors
				idxs = list((1-neighbors).nonzero())
				idxs[0] += max(row-1, min_row)
				idxs[1] += max(col-1, min_col)
				idxs = list(zip(idxs[0], idxs[1]))

				# Randomly pick a non-zero neighbor
				next_row, next_col = choice(idxs)

				# Add current point to list of traversed points
				# and move to the new point
				start = (next_row, next_col)
				self.image[row,col] = 0
				self.image[next_row, next_col] = 1

	# Run DLA
	def run(self, num_iterations=None, stickiness=1.0):

		'''
			Method to run a DLA simulation. This consists of two steps:
				1. Add a particle randomly to the system.
				2. Let the particle do a random walk until it gets
				   stuck to any of the existing particles.
		'''

		# Assert constraints on inputs
		assert (num_iterations >= 0) and (type(num_iterations)==int), "num_points can only be an integer greater than or equal to 0"
		assert 0.0<stickiness<=1.0, "stickiness can only be in the interval (0.0, 1.0]"

		# Set number of iterations 
		if num_iterations is None:
			num_iterations = self.num_points

		# Add particles into the system one-by-one and run random walks
		# until the particle gets stuck
		for _ in tqdm(range(num_iterations), total=num_iterations):

			# Add particle to the system
			px, py = self._add_particle()

			# Run random walk until the particle gets stuck
			self._run_random_walk(
				start=(px,py),
				stickiness=stickiness)
