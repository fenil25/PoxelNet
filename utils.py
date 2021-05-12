import trimesh
from matplotlib import pyplot as plt
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F

# display the mesh
def show_mesh(path):
    mesh = trimesh.load(path)
    return mesh.show()

# display the point cloud sampled from the mesh
def show_point_cloud(path, N = 2048):
    mesh = trimesh.load(path)
    points = mesh.sample(N)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()

# display points of the point cloud
def display_points(points):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()

# Collating batches into Minkowski sparse tensor 
def collate_function(batch_data):
    coordinates, features, labels = [], [], []
    for batch in batch_data:
        coordinates.append(batch["coordinates"])
        features.append(batch["features"])
        labels.append(batch["labels"])

    coordinates, features, labels = ME.utils.sparse_collate(coordinates, features, labels, dtype=torch.float32)
    
    return {
        "coordinates": coordinates,
        "features": features,
        "labels": torch.tensor(labels, dtype=torch.int64)
    }

# Creating a batch for training
def create_input_batch(batch, device="cpu", quantization_size=0.05):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(coordinates=batch["coordinates"], features=batch["features"], device=device)

#  Calculating cross entropy loss and label smoothing
def criterion(pred, labels, smoothing=True):
    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")

    return loss