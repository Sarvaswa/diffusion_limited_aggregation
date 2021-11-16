import os
import csv
import numpy as np
import pandas as pd
from DLA import DiffusionLimitedAggregation as dla


if __name__ == "__main__":

	# Define desired range for image size, number of particles, and stickiness
	shape_range = list(range(251, 1052, 200))
	iter_range = list(range(2000, 20000, 2500))[::-1]
	stickiness_range = ([num/100 for num in range(5,100,10)]+[1.0])[::-1]

	# Define required directories
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(curr_dir, "images/")
	output = os.path.join(curr_dir, "data.csv")
	df_cols = ["num_particles", "stickiness", "filepath"]
	output_df = pd.DataFrame(columns=df_cols)

	# Run DLA for the specified parameter range
	for shape in shape_range:
		for iter_val in iter_range:
			for stickiness in stickiness_range:

				# Run DLA
				print(f"\n Current Config: SHAPE={shape}, N={iter_val}, Stickiness={stickiness}")
				d = dla(shape=(shape,shape))
				d.run(num_iterations=iter_val, stickiness=stickiness)

				# Save image a numpy array
				filename = f"dla_shape{shape}_N{iter_val}_stick{stickiness}.npy"
				filepath = os.path.join(image_dir, filename)
				np.save(filepath, d.image)

				# Add data to CSV
				curr_data = [iter_val, stickiness, filepath]
				curr_df = pd.DataFrame([curr_data], columns=df_cols)
				output_df = output_df.append(curr_df)

				# Save output dataframe to disk as CSV
				out_filepath = os.path.join(curr_dir, "output_data.csv")
				output_df.to_csv(out_filepath)
