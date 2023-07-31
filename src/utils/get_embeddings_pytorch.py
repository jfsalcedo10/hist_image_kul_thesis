import torch
import numpy as np

def get_embeddings_pytorch(model, dataloader, device):
    """Generate embeddings for a PyTorch model used for 
    another supervised task. Built for labeled datasets.

    Args:
        model: PyTorch model
        dataloader: Dataloader of the dataset to generate embeddings for
        device: Device where the operation is performed (i.e. GPU or CPU)

    Returns:
        The embeddings, and labels for the model
    """
    model.eval()
    embeddings = []
    labels = []

    for data, labels_batch in dataloader:
        images = data.to(device)

        with torch.no_grad():
            output = model(images)
        
        embeddings.append(output.detach().cpu().numpy())
        labels.append(labels_batch.numpy())

    embeddings = torch.tensor(np.concatenate(embeddings), dtype= torch.float)
    labels = torch.tensor(np.concatenate(labels), dtype=torch.float).reshape(-1,1)

    return embeddings, labels