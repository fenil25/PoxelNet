import os
import numpy as np
import argparse
import trimesh

parser = argparse.ArgumentParser(description='Sample Points from Meshes and Create Point Cloud Dataset')
parser.add_argument('--path', type=str, help='Path to the dataset', required = True)
parser.add_argument('--N', type=int, help='Number of points to sample from each mesh', default=4096)

args = parser.parse_args()

map = {}

train_points, train_labels = [], []
test_points, test_labels = [], []

DIR_PATH = args.path

for idx, class_name in enumerate(os.listdir(DIR_PATH)):
	map[idx] = class_name
	train_class_path = os.path.join(DIR_PATH, class_name, "train")
	test_class_path = os.path.join(DIR_PATH, class_name, "test")
	print(f"Creating training set for class {class_name}")
	for file_name in os.listdir(train_class_path):
		train_file_name = os.path.join(train_class_path, file_name)
		mesh = trimesh.load(train_file_name)
		points = mesh.sample(args.N).astype('float32')
		train_points.append(points)
		train_labels.append(idx)
	print(f"Creating testing set for class {class_name}")
	for file_name in os.listdir(test_class_path):
		test_file_name = os.path.join(test_class_path, file_name)
		mesh = trimesh.load(test_file_name)
		points = mesh.sample(args.N).astype('float32')
		test_points.append(points)
		test_labels.append(idx)

train_dict = {
	"points": np.array(train_points),
	"labels": np.array(train_labels),
	"map": map
}

test_dict = {
	"points": np.array(test_points),
	"labels": np.array(test_labels),
	"map": map
}

np.save("train.npy", train_dict)
np.save("test.npy", test_dict)