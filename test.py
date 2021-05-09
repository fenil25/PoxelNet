import torch
import sklearn.metrics as metrics

from modelnet40_dataset import ModelNet40
from model import PoxelNet
from utils import collate_function, create_input_batch

from tqdm import tqdm
import argparse
import numpy as np

def test(net, test_dataset, device, args):
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn=collate_function)

    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input = create_input_batch(batch, device=device, quantization_size=args.voxel_size,)
            logit = net(input)
            pred = torch.argmax(logit, 1)
            temp = batch["labels"]
            labels.append(temp.cpu().numpy())
            preds.append(pred.cpu().numpy())
            torch.cuda.empty_cache()

    return metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the model')
    parser.add_argument("--voxel_size", type=float, default=0.05, help="More the vocel size, lesser the resolution")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size of dataloader")
    parser.add_argument("--path_to_dataset", type=str, help="Path to the training and testing points directory", required=True)

    parser.add_argument("--seed", type=int, default=777)

    parser.add_argument("--path_weights", type=str, default="modelnet.pth", help="Path to the trained weights")

    args = parser.parse_args()
    print(args)

    print("Loading the test dataset...")
    test_dataset = ModelNet40(args.path_to_dataset, phase = "test", transform = None)
    print("Dataset successfully loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = PoxelNet(in_channel = 3, out_channel = 40, embedding_channel = 1024).to(device)
    checkpoint = torch.load(args.path_weights)
    net.load_state_dict(checkpoint['state_dict'])
    print("Trained PoxelNet model loaded")

    accuracy = test(net, test_dataset, device, args)
    print(f"Accuracy of the model is : {accuracy}")