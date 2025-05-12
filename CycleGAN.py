import os
import csv
import argparse
from tqdm import tqdm
from PIL import Image, ImageFilter
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision import models

# -----------------------------------------------------------------------------
# Common Transform
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# -----------------------------------------------------------------------------
# Unpaired Dataset for Both Domains
# -----------------------------------------------------------------------------
class UnpairedFrameDataset(Dataset):
    """
    Dataset class for unpaired training.
    Expects a CSV file with columns: frame_path, edge_path, depth_path, color_path, prev_frame_path.
    The dataset returns:
      - condition_maps (8-channel tensor):
          [edge (1), depth (1), color (3), previous_frame (3)]
      - frame image (3 channels) as target (used only for adversarial and cycle losses)
    The same dataset structure is used for both domains (rendered and anime),
    potentially with different CSV files.
    """

    def __init__(self, csv_file, transform=None, blur_radius=5, domain="anime"):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.blur_radius = blur_radius
        self.domain = domain  # "anime" or "render"

    def __len__(self):
        return len(self.data)

    def _load_image(self, path, mode):
        return Image.open(path).convert(mode)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Load the image (the "frame")
        frame = self._load_image(row['frame_path'], 'RGB')
        # Load condition maps.
        edge = self._load_image(row['edge_path'], 'L')
        depth = self._load_image(row['depth_path'], 'L')
        color = self._load_image(row['color_path'], 'RGB')
        prev_frame = self._load_image(row['prev_frame_path'], 'RGB')
        # To discourage direct copying, we blur the previous frame.
        prev_frame = prev_frame.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.transform:
            frame = self.transform(frame)
            edge = self.transform(edge)
            depth = self.transform(depth)
            color = self.transform(color)
            prev_frame = self.transform(prev_frame)

        # Concatenate: edge (1 channel), depth (1), color (3), previous frame (3) = 8 channels.
        condition_maps = torch.cat([edge, depth, color, prev_frame], dim=0)
        return condition_maps, frame


# -----------------------------------------------------------------------------
# Direct U-Net Model (used as Generator architecture)
# -----------------------------------------------------------------------------
class DirectUNet(nn.Module):
    """
    U-Net style encoder-decoder that takes an 8-channel input (condition maps)
    and outputs a 3-channel image.
    """

    def __init__(self, in_channels=8, out_channels=3, features=[64, 128, 256, 512]):
        super(DirectUNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

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
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_layers[idx](x)
        return self.final_conv(x)


# -----------------------------------------------------------------------------
# PatchGAN Discriminator
# -----------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(PatchDiscriminator, self).__init__()
        layers = []
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# VGG Feature Extractor for Optional Perceptual/Style Losses (if desired)
# -----------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=None, use_input_norm=True, device='cuda'):
        super(VGGFeatureExtractor, self).__init__()
        self.device = device
        if layers is None:
            layers = {'3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"}
        self.layers = layers
        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
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
    (b, ch, h, w) = feature.size()
    feat = feature.view(b, ch, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (ch * h * w)


# -----------------------------------------------------------------------------
# CycleGAN Training Function for Unpaired Translation
# -----------------------------------------------------------------------------
def train_cyclegan_unpaired(render_csv, anime_csv, epochs=100, batch_size=4, lr=2e-4,
                            device='cuda', lambda_cycle=10.0, lambda_id=5.0, lambda_adv=1.0):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Data loaders for rendered and anime domains.
    rendered_dataset = UnpairedFrameDataset(render_csv, transform=transform, blur_radius=5, domain="render")
    anime_dataset = UnpairedFrameDataset(anime_csv, transform=transform, blur_radius=5, domain="anime")

    rendered_loader = DataLoader(rendered_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    anime_loader = DataLoader(anime_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Initialize generators:
    # G: Render -> Anime. This is initialized with pre-trained U-Net weights from Stage 1.
    G = DirectUNet(in_channels=8, out_channels=3).to(device)
    pretrained_path = "direct_unet_temporal_model.pth"
    if os.path.exists(pretrained_path):
        G.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Loaded pre-trained U-Net weights into G.")
    else:
        print("Pre-trained U-Net weights not found. G is randomly initialized.")

    # F: Anime -> Render (randomly initialized).
    F = DirectUNet(in_channels=8, out_channels=3).to(device)

    # Discriminators:
    D_A = PatchDiscriminator(in_channels=3).to(device)  # For anime domain.
    D_R = PatchDiscriminator(in_channels=3).to(device)  # For rendered domain.

    # Loss functions.
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_cycle = nn.L1Loss()
    criterion_id = nn.L1Loss()

    # Optionally, you can also set up perceptual/style losses using VGGFeatureExtractor.
    # For simplicity in this implementation they are omitted, but you can add them as extra loss terms.

    optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_R = optim.Adam(D_R.parameters(), lr=lr, betas=(0.5, 0.999))

    # Create iterators for the dataloaders.
    anime_iter = iter(anime_loader)
    rendered_iter = iter(rendered_loader)
    num_batches = min(len(anime_loader), len(rendered_loader))

    for epoch in range(epochs):
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in pbar:
            # Get batches; if one iterator runs out, reinitialize.
            try:
                condition_R, real_R = next(rendered_iter)
            except StopIteration:
                rendered_iter = iter(rendered_loader)
                condition_R, real_R = next(rendered_iter)
            try:
                condition_A, real_A = next(anime_iter)
            except StopIteration:
                anime_iter = iter(anime_loader)
                condition_A, real_A = next(anime_iter)

            condition_R = condition_R.to(device)
            real_R = real_R.to(device)
            condition_A = condition_A.to(device)
            real_A = real_A.to(device)

            # Generate fake images.
            fake_A = G(condition_R)  # Render -> Anime (fake anime)
            fake_R = F(condition_A)  # Anime -> Render (fake render)

            # Cycle: Render -> Anime -> Render.
            rec_R = F(
                torch.cat([real_A[:, 0:2, :, :], real_A[:, 2:5, :, :], real_A[:, 5:8, :, :]], dim=1)) if False else F(
                condition_R)
            # For simplicity, we use the original condition map again.
            # Alternatively, you can re-extract condition maps from fake images.
            rec_R = F(condition_R)  # Cycle consistency on rendered side.

            # Cycle: Anime -> Render -> Anime.
            rec_A = G(condition_A)

            # Identity mapping:
            id_A = G(condition_A)
            id_R = F(condition_R)

            # Labels for adversarial loss.
            valid = torch.ones_like(D_A(real_A)).to(device)
            fake_label = torch.zeros_like(D_A(real_A)).to(device)

            # ---------------------
            #  Train Generators G and F
            # ---------------------
            optimizer_G.zero_grad()

            # Adversarial losses.
            loss_G_adv = criterion_GAN(D_A(fake_A), valid)
            loss_F_adv = criterion_GAN(D_R(fake_R), valid)

            # Cycle consistency losses.
            loss_cycle_R = criterion_cycle(rec_R, real_R)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            loss_cycle = loss_cycle_R + loss_cycle_A

            # Identity losses.
            loss_id_A = criterion_id(id_A, real_A)
            loss_id_R = criterion_id(id_R, real_R)
            loss_id = loss_id_A + loss_id_R

            # Total generator loss.
            loss_G_total = (lambda_adv * (loss_G_adv + loss_F_adv) +
                            lambda_cycle * loss_cycle +
                            lambda_id * loss_id)

            loss_G_total.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator D_A (anime domain)
            # ---------------------
            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_GAN(D_A(real_A), valid)
            loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), fake_label)
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------------------
            #  Train Discriminator D_R (rendered domain)
            # ---------------------
            optimizer_D_R.zero_grad()
            loss_D_R_real = criterion_GAN(D_R(real_R), valid)
            loss_D_R_fake = criterion_GAN(D_R(fake_R.detach()), fake_label)
            loss_D_R = 0.5 * (loss_D_R_real + loss_D_R_fake)
            loss_D_R.backward()
            optimizer_D_R.step()

            pbar.set_postfix({
                "G_adv": f"{(loss_G_adv + loss_F_adv).item():.3f}",
                "cycle": f"{loss_cycle.item():.3f}",
                "id": f"{loss_id.item():.3f}",
                "D_A": f"{loss_D_A.item():.3f}",
                "D_R": f"{loss_D_R.item():.3f}"
            })

        # Optionally, save some sample outputs each epoch.
        sample_save_path = os.path.join("samples", f"epoch_{epoch + 1}.png")
        os.makedirs("samples", exist_ok=True)
        # Save a composite image using the first sample from rendered domain.
        condition_sample, _ = rendered_dataset[0]
        condition_sample = condition_sample.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_A_sample = G(condition_sample)
        # Here we assume that channels 2-4 in the condition represent a color map.
        cond_color = condition_sample[0, 2:5, :, :].cpu()
        fake_A_img = to_pil_image(torch.clamp(fake_A_sample.squeeze(0).cpu(), 0, 1))
        cond_color_img = to_pil_image(cond_color)
        composite = Image.new("RGB", (cond_color_img.width * 2, cond_color_img.height))
        composite.paste(cond_color_img, (0, 0))
        composite.paste(fake_A_img, (cond_color_img.width, 0))
        composite.save(sample_save_path)
        print(f"Saved sample composite image to {sample_save_path}")

        if (epoch + 1) % 5 == 0:
            torch.save(G.state_dict(), f"cyclegan_G_render2anime_epoch_{epoch + 1}.pth")
            torch.save(F.state_dict(), f"cyclegan_F_anime2render_epoch_{epoch + 1}.pth")
            torch.save(D_A.state_dict(), f"cyclegan_D_A_epoch_{epoch + 1}.pth")
            torch.save(D_R.state_dict(), f"cyclegan_D_R_epoch_{epoch + 1}.pth")
            print(f"Saved models at epoch {epoch + 1}")

    # Save final models.
    torch.save(G.state_dict(), "cyclegan_G_render2anime.pth")
    torch.save(F.state_dict(), "cyclegan_F_anime2render.pth")
    print("CycleGAN training complete and models saved.")


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CycleGAN Unpaired Training for Rendered to Anime Translation using Pre-trained U-Net as Stage 1 (Generator G)"
    )
    parser.add_argument("--render_csv", type=str, required=True, help="Path to CSV file for rendered frames")
    parser.add_argument("--anime_csv", type=str, required=True, help="Path to CSV file for anime frames")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="Identity loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Adversarial loss weight")
    args = parser.parse_args()

    train_cyclegan_unpaired(
        render_csv=args.render_csv,
        anime_csv=args.anime_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        lambda_cycle=args.lambda_cycle,
        lambda_id=args.lambda_id,
        lambda_adv=args.lambda_adv
    )