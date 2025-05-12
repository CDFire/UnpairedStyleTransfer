import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


# -----------------------------------------------------------------------------
# U-Net Model Definition (Same as used for training)
# -----------------------------------------------------------------------------
class DirectUNet(nn.Module):
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
# Video Generation Function Using CSV Input
# -----------------------------------------------------------------------------
def generate_video_from_csv(csv_path, model_path, output_video, device='cpu', fps=10):
    # Define image preprocessing (same as during training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load the CSV file and sort by filename (or any column that ensures the correct order)
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="filename")
    num_frames = len(df)
    if num_frames == 0:
        raise ValueError("No samples found in the CSV file.")

    # Set up the model
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = DirectUNet(in_channels=8, out_channels=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Setup OpenCV video writer (assuming 256x256 frames)
    frame_width, frame_height = 256, 256
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize previous frame as a zero tensor
    prev_tensor = torch.zeros((3, frame_height, frame_width), device=device)

    print(f"Generating video from {num_frames} samples...")

    with torch.no_grad():
        for idx, row in df.iterrows():
            # Get file paths from the CSV row
            # Note that frame_path is present in CSV but not used for inference in this example.
            edge_path = row["edge_path"]
            depth_path = row["depth_path"]
            color_path = row["color_path"]

            # Load and transform each condition map
            edge_img = Image.open(edge_path).convert("L")
            depth_img = Image.open(depth_path).convert("L")
            color_img = Image.open(color_path).convert("RGB")

            edge_tensor = transform(edge_img)  # shape: [1, 256, 256]
            depth_tensor = transform(depth_img)  # shape: [1, 256, 256]
            color_tensor = transform(color_img)  # shape: [3, 256, 256]

            # Concatenate to create an 8-channel input: [edge (1) + depth (1) + color (3) + previous frame (3)]
            condition_map = torch.cat([edge_tensor, depth_tensor, color_tensor, prev_tensor.cpu()], dim=0)
            condition_map = condition_map.unsqueeze(0).to(device)

            # Inference with the model
            output = model(condition_map)
            output = torch.clamp(output, 0, 1)

            # Update previous frame with the current output
            prev_tensor = output.squeeze(0)

            # Convert output tensor to PIL image then to NumPy (BGR) for video writing
            out_img = to_pil_image(prev_tensor.cpu())
            out_np = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
            video_writer.write(out_np)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_frames} frames")

    video_writer.release()
    print(f"Video saved to {output_video}")


# -----------------------------------------------------------------------------
# Main: Command-Line Interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a render dataset stored in a CSV file. "
                    "Each sample uses the previously generated frame as the input previous frame."
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the CSV file with columns: filename,frame_path,edge_path,depth_path,color_path")
    parser.add_argument("--model_path", type=str, default="direct_unet_temporal_model.pth",
                        help="Path to the trained U-Net model weights.")
    parser.add_argument("--output_video", type=str, default="output_video.mp4",
                        help="Path to save the generated video.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on ('cpu' or 'cuda').")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    args = parser.parse_args()

    generate_video_from_csv(args.csv, args.model_path, args.output_video, args.device, args.fps)