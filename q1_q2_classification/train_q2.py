import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Subset


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        # Load pretrained ResNet18 on ImageNet
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.resnet.fc.in_features

        # Replace last layer for Pascal VOC (20 classes)
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


def visualize_tsne(model_path='best_resnet18.pth', data_dir='data/VOCdevkit/VOC2007', size=224):
    """Visualize 2D t-SNE of feature embeddings for PASCAL VOC."""
    from torchvision import models

    print("Loading dataset...")
    test_dataset = VOCDataset(split='test', size=size, data_dir=data_dir)
    indices = random.sample(range(len(test_dataset)), 1000)
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    print("Loading trained ResNet18...")
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    # Strip "resnet." prefix from keys in state_dict if present
    state_dict = {k.replace('resnet.', ''): v for k, v in checkpoint.items()}

    model = models.resnet18(weights=None)
    model.fc = nn.Identity()

    # Load weights safely
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    # Collect features and labels
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.cuda()
            feats = model(imgs).cpu().numpy()
            features.append(feats)
            labels.append(lbls.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    print("Running t-SNE projection (this may take a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init='pca', random_state=0)
    features_2d = tsne.fit_transform(features)

    # Assign colors for 20 Pascal classes
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = np.array([base_colors[i % len(base_colors)] for i in range(20)])

    def get_color_for_label(label_vec):
        active = np.where(label_vec == 1)[0]
        if len(active) == 0:
            return np.array([0.5, 0.5, 0.5])  # gray for background
        return np.mean([mcolors.to_rgb(colors[i]) for i in active], axis=0)

    point_colors = np.array([get_color_for_label(lbl) for lbl in labels])

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, s=10)
    plt.title('t-SNE Visualization of PASCAL Test Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Legend mapping from color to class name
    for i, cname in enumerate(VOCDataset.CLASS_NAMES):
        plt.scatter([], [], color=colors[i], label=cname)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('tsne_pascal.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    args = ARGS(
        epochs=20,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=1e-4,
        batch_size=32,
        step_size=15,
        gamma=0.5
    )

    print(args)

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train the model
    # test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    # print('test map:', test_map)

    # Optional: visualize t-SNE after training
    # Save checkpoint as best_resnet18.pth first, then:
    visualize_tsne('best_resnet18.pth')
