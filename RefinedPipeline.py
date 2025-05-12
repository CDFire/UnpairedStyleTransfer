import os
import csv
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision import models
from torchvision.models import VGG16_Weights

# -----------------------------------------------------------------------------
# Common Transform
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# -----------------------------------------------------------------------------
# Dataset Definitions (Modified for Refined Temporal Consistency)
# -----------------------------------------------------------------------------
class AnimeDataset(Dataset):
    """
    For supervised training with temporal consistency.
    CSV columns: filename, prev_frame_path, frame_path, edge_path, depth_path, color_path
    Loads:
      - Condition maps (edge, depth, color) concatenated into a 5‑channel tensor.
      - Previous anime frame is processed with a Gaussian blur to remove high‐frequency details,
        reducing the chance of “cheating” while still providing temporal context.
      - Target anime frame.
    """

    def __init__(self, csv_file, transform=None, blur_radius=5):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.blur_radius = blur_radius

    def __len__(self):
        return len(self.data)

    def _load_image(self, path, mode):
        return Image.open(path).convert(mode)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Load target anime frame (RGB)
        target = self._load_image(row['frame_path'], 'RGB')
        # Load condition maps:
        edge = self._load_image(row['edge_path'], 'L')
        depth = self._load_image(row['depth_path'], 'L')
        color = self._load_image(row['color_path'], 'RGB')
        # Load previous frame (RGB) and blur it to avoid direct detail copying.
        prev_frame = self._load_image(row['prev_frame_path'], 'RGB')
        prev_frame = prev_frame.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.transform:
            target = self.transform(target)
            edge = self.transform(edge)
            depth = self.transform(depth)
            color = self.transform(color)
            prev_frame = self.transform(prev_frame)

        # Concatenate condition maps:
        # edge (1 channel), depth (1 channel), color (3 channels) => 5 channels.
        # Append the blurred previous frame (3 channels) for temporal consistency.
        condition_maps = torch.cat([edge, depth, color, prev_frame], dim=0)
        return condition_maps, target


# -----------------------------------------------------------------------------
# Direct Supervised U-Net Model with Temporal Consistency
# -----------------------------------------------------------------------------
class DirectUNet(nn.Module):
    """
    U-Net style encoder-decoder network that takes an 8‑channel tensor (5 channels from condition maps
    concatenated with 3 channels from the blurred previous frame) as input and outputs the target original image (3 channels).
    """

    def __init__(self, in_channels=8, out_channels=3, features=[64, 128, 256, 512]):
        super(DirectUNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: sequential downsampling layers with skip connections.
        channels = in_channels
        for feature in features:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            channels = feature

        # Bottleneck layer.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: upsampling with skip connections.
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        rev_features = features[::-1]
        channels = features[-1] * 2
        for feature in rev_features:
            self.upconvs.append(
                nn.ConvTranspose2d(channels, feature, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            channels = feature
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder_layers:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            # Adjust dimensions if necessary.
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_layers[idx](x)
        return self.final_conv(x)


# -----------------------------------------------------------------------------
# VGG Feature Extractor for Perceptual and Style Losses
# -----------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=None, use_input_norm=True, device='cuda'):
        super(VGGFeatureExtractor, self).__init__()
        self.device = device
        # Specify layers to extract: keys are indices of feature layers,
        # and values are human-readable names.
        if layers is None:
            layers = {'3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"}
        self.layers = layers
        # Load pre-trained VGG16 and freeze its parameters.
        weights = VGG16_Weights.IMAGENET1K_V1
        self.vgg = models.vgg16(weights=weights).features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # VGG expects normalized input in a specific range.
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        self.use_input_norm = use_input_norm

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features


def gram_matrix(feature):
    # Compute the Gram matrix for a given feature map.
    (b, ch, h, w) = feature.size()
    feat = feature.view(b, ch, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (ch * h * w)


# -----------------------------------------------------------------------------
# Helper Function to Save a Composite Sample Image
# -----------------------------------------------------------------------------
def save_epoch_sample(epoch, condition_maps, target, output, save_dir="samples"):
    """
    Creates a composite image showing:
      - Top-left: The RGB portion of the condition maps (channels 2-4, i.e. the color map)
      - Top-right: The blurred previous frame (channels 6-8)
      - Bottom-left: The target image (desired output)
      - Bottom-right: The model output (predicted)
    Saves the composite image to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    # condition_maps is an 8-channel tensor. Extract:
    # Channels 2-4 correspond to the color map (input condition's color portion)
    cond_color = condition_maps[2:5, :, :]
    # Channels 5-7 (index 5,6,7) correspond to the blurred previous frame.
    prev_frame = condition_maps[5:8, :, :]

    # Convert tensors to PIL images.
    cond_color_img = to_pil_image(cond_color.cpu())
    prev_frame_img = to_pil_image(prev_frame.cpu())
    target_img = to_pil_image(target.cpu())
    output_img = to_pil_image(output.cpu())

    # Create a composite image in a 2x2 grid.
    width, height = cond_color_img.size
    composite = Image.new("RGB", (width * 2, height * 2))
    composite.paste(cond_color_img, (0, 0))
    composite.paste(prev_frame_img, (width, 0))
    composite.paste(target_img, (0, height))
    composite.paste(output_img, (width, height))

    composite_path = os.path.join(save_dir, f"epoch_{epoch + 1}_sample.png")
    composite.save(composite_path)
    print(f"Saved sample visualization for epoch {epoch + 1} to {composite_path}")


# -----------------------------------------------------------------------------
# Training Function for Direct Supervised Learning with Composite Losses
# -----------------------------------------------------------------------------
def train_direct_supervised(csv_file, epochs=10, batch_size=4, lr=1e-4, device='cuda',
                            pixel_weight=1.0, perceptual_weight=0.1, style_weight=1.0):
    # Initialize dataset and dataloader.
    dataset = AnimeDataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Initialize the U-Net model.
    model = DirectUNet(in_channels=8, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Using L1 loss for pixel-wise regression.
    pixel_loss_fn = nn.L1Loss()

    # Initialize the VGG feature extractor (for perceptual and style losses).
    vgg_extractor = VGGFeatureExtractor(device=device)

    model.train()
    for epoch in range(epochs):
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for condition_maps, target in epoch_bar:
            condition_maps = condition_maps.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(condition_maps)
            output = torch.clamp(output, 0, 1)

            # Compute pixel (L1) loss.
            loss_pixel = pixel_loss_fn(output, target)

            # Compute VGG features for perceptual and style losses.
            features_output = vgg_extractor(output)
            features_target = vgg_extractor(target)

            loss_perceptual = 0.0
            loss_style = 0.0
            for key in features_output:
                # Perceptual loss: feature-wise L1 difference.
                loss_perceptual += F.l1_loss(features_output[key], features_target[key])
                # Style loss: L1 difference between Gram matrices.
                gram_out = gram_matrix(features_output[key])
                gram_target = gram_matrix(features_target[key])
                loss_style += F.l1_loss(gram_out, gram_target)

            # Total loss: weighted sum of pixel, perceptual, and style losses.
            total_loss = (pixel_weight * loss_pixel +
                          perceptual_weight * loss_perceptual +
                          style_weight * loss_style)

            total_loss.backward()
            optimizer.step()
            epoch_bar.set_postfix({"Total Loss": total_loss.item(), "Pixel": loss_pixel.item(),
                                   "Perceptual": loss_perceptual.item(), "Style": loss_style.item()})

        # Save a composite sample for visualization.
        model.eval()
        with torch.no_grad():
            sample_condition_maps, sample_target = dataset[0]
            sample_condition_maps = sample_condition_maps.unsqueeze(0).to(device)
            sample_output = model(sample_condition_maps)
            sample_output = torch.clamp(sample_output, 0, 1)
            sample_condition_maps = sample_condition_maps.squeeze(0)
            sample_target = sample_target.cpu()
            sample_output = sample_output.squeeze(0).cpu()
            save_epoch_sample(epoch, sample_condition_maps, sample_target, sample_output)
        model.train()

    torch.save(model.state_dict(), "old models/direct_unet_temporal_model.pth")
    print("Direct supervised training complete and model saved.")


# -----------------------------------------------------------------------------
# Main: Command-Line Interface
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Direct Supervised U-Net Training with Temporal Consistency and Composite Losses"
    )
    parser.add_argument("--stage", type=str, choices=["direct_supervised"], required=True,
                        help="Stage to run: direct_supervised")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing paired examples")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    # Loss hyperparameters
    parser.add_argument("--pixel_weight", type=float, default=1.0, help="Weight for pixel loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--style_weight", type=float, default=1.0, help="Weight for style loss")
    args = parser.parse_args()

    if args.stage == "direct_supervised":
        if not args.csv:
            print("CSV file is required for supervised training")
        else:
            train_direct_supervised(args.csv, epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, device=args.device,
                                    pixel_weight=args.pixel_weight,
                                    perceptual_weight=args.perceptual_weight,
                                    style_weight=args.style_weight)