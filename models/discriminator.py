import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        """
        Vanilla Fully-Connected Discriminator for MNIST.
        Uses Spectral Normalization for Lipschitz continuity.
        
        Args:
            img_dim (int): Flattened dimension of MNIST (1x28x28 = 784).
        """
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: 784 -> 1024
            nn.utils.spectral_norm((nn.Linear(img_dim, 1024))),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 1024 -> 512
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 512 -> 256
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 -> 1
            nn.utils.spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, img):
        """
        Args:
            img: Tensor of shape (B, 1, 28, 28) or (B, 784).
        Returns:
            logits: Tensor of shape (B, 1) containing raw scores/logits.
        """
        # Flatten the image: (B, 1, 28, 28) -> (B, 784)
        img_flat = img.view(img.size(0), -1) 
        
        # Forward pass
        logits = self.model(img_flat)
        
        return logits

if __name__ == "__main__":
    # Settings
    img_dim = 784
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

    print(f"Testing Discriminator on {device}...")

    # Instantiate discriminator
    disc = Discriminator(img_dim=img_dim).to(device)
    
    # Create dummy input
    fake_img = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Forward pass
    score = disc(fake_img)
    
    # 4. Verification
    print(f"Input image shape: {fake_img.shape}")
    print(f"Output score shape: {score.shape}")
    
    # Assertions
    assert score.shape == (batch_size, 1), f"Error: Expected shape {(batch_size, 1)}, got {score.shape}"
    
    print("Discriminator test passed!")