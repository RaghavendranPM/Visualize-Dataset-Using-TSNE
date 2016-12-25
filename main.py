import data_visualization as dv
import numpy as np

# Removed last row from battles.csv since a majority of important fields were missing
label_names = ["attacker_outcome", "major_death", "major_capture"]
test_percent = 0.99 # 99% test since the data size is quite small

for label in label_names:
	battles_x_train, battles_x_test, battles_y_train, battles_y_test = dv.get_data("battles.csv", label, test_percent)
	battles_x_train = np.delete(battles_x_train, [0, 1, 2], axis=1)
	battles_x_test = np.delete(battles_x_test, [0, 1, 2], axis=1)
	print battles_x_test
	print battles_y_test

	dv.perform_tsne(battles_x_test, battles_y_test)