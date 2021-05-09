import torch
import numpy as np
import torch.optim as optim

from modelnet40_dataset import ModelNet40
from transformations import CoordinateTransformation
from model import PoxelNet
from utils import criterion, collate_function, create_input_batch
from test import test

import argparse


def train(net, dataset, test_dataset, device, args):
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    curr_iter = 0

    if args.load_pretrained:
      print('Loading model from saved state...')
      checkpoint = torch.load(args.path_weights)
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      curr_iter = checkpoint['curr_iter']

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function)
    train_iter = iter(dataloader)
    best_metric = 0
    net.train()

    print("Starting the training from scratch")

    for i in range(curr_iter, curr_iter+args.max_steps):
        optimizer.zero_grad()
        
        try:
            data_dict = train_iter.next()
        except StopIteration:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, collate_fn=collate_function)
            train_iter = iter(dataloader)
            data_dict = train_iter.next()

        input = create_input_batch(data_dict, device=device, quantization_size=args.voxel_size)

        temp = data_dict["labels"]
        logit = net(input)
        loss = criterion(logit, temp.to(device))

        loss.backward()

        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()

        if i % args.stat_freq == 0:
            print(f"Iter: {i}, Loss: {loss.item():.3e}")

        if i % args.val_freq == 0 and i > 0:

            accuracy = test(net, test_dataset, device, args)
            if best_metric < accuracy:
                best_metric = accuracy
                best_vals = {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                }

            torch.save(best_vals, args.store_weights)
            print(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training the model')
	parser.add_argument("--voxel_size", type=float, default=0.05, help="More the vocel size, lesser the resolution")
	parser.add_argument("--max_steps", type=int, default=15000, help="Number of training steps")
	parser.add_argument("--val_freq", type=int, default=1000, help="Frequency of testing on validation set")
	parser.add_argument("--batch_size", default=32, type=int, help="Batch size of dataloader")
	parser.add_argument("--lr", default=1e-1, type=float, help="Learning rate of SGD optimizer")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")

	parser.add_argument("--path_to_dataset", type=str, help="Path to the training and testing points directory", required=True)
	parser.add_argument("--store_weights", type=str, default="modelnet.pth", help="Location where weights would be stored")

	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--stat_freq", type=int, default=100)
	parser.add_argument("--seed", type=int, default=777)

	parser.add_argument("--load_pretrained", action="store_true", help="Load the pretrained weights and start training from there")
	parser.add_argument("--path_weights", type=str, default="modelnet.pth", help="Path to trained weights for resuming training")

	args = parser.parse_args()
	print(args)

	ctr = CoordinateTransformation()

	print("Loading the ModelNet40 dataset...")
	train_dataset = ModelNet40(args.path_to_dataset, phase = "train", transform = ctr)
	test_dataset = ModelNet40(args.path_to_dataset, phase = "test", transform = None)
	print("Dataset successfully loaded")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = PoxelNet(in_channel = 3, out_channel = 40, embedding_channel = 1024).to(device)

	train(net, train_dataset, test_dataset, device, args)