import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        """
        Vanilla Fully-Connected Generator for MNIST.
        Includes Batch Normalization for training stability.
        
        Args:
            z_dim (int): Dimension of the latent noise vector (default: 100).
            img_dim (int): Flattened dimension of MNIST (1x28x28 = 784).
        """
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: 100 -> 256
            nn.Linear (z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 256 -> 512
            nn.Linear (256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 512 -> 1024
            nn.Linear (512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 1024 -> 784
            nn.Linear (1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, z_dim)
        Returns:
            img: Tensor of shape (batch_size, 1, 28, 28)
        """
        # Forward pass
        img_flat = self.model(z)

        # Reshape the output: (B, 784) -> (B, 1, 28, 28)
        img = img_flat.view(img_flat.shape[0], 1, 28, 28)
        return img
    

if __name__ == "__main__":
    # Settings
    z_dim = 100
    batch_size = 5
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal).")
    else:
        device = torch.device("cpu")
        print("Using device: CPU.")

    print(f"Testing Generator on {device}...")
    
    # Instantiate generator
    gen = Generator(z_dim=z_dim).to(device)
    
    # Create dummy input
    z = torch.randn(batch_size, z_dim).to(device)
    
    # Forward pass
    img = gen(z)
    
    # Verification
    print(f"Input z shape:  {z.shape}")
    print(f"Output img shape: {img.shape}")
    
    # Assertions
    assert img.shape == (batch_size, 1, 28, 28), "Error: Incorrect output shape"
    assert img.max() <= 1.0 and img.min() >= -1.0, "Error: Output values out of range [-1, 1]"
    
    print("Generator test passed!")