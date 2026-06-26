"""
Model definitions for MNIST classification.

This module contains three different architectures:
  1. MLP  (Multi-Layer Perceptron)  - simplest, ~97% accuracy
  2. CNN  (Convolutional Neural Net) - standard for images, ~99% accuracy
  3. ViT  (Vision Transformer)      - modern attention-based, ~98% accuracy

Each model:
  - Inherits from nn.Module (PyTorch's base class for all neural networks).
  - Overrides __init__() to define layers.
  - Overrides forward() to define how data flows through those layers.
  - Accepts input shape: (batch_size, 1, 28, 28)  =  (B, C, H, W)
  - Returns output shape: (batch_size, 10)         =  scores for digits 0-9

The final build_model() factory lets you choose which one by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (a.k.a. "fully connected network").

    --- How it works ---
    1. Flatten the 2D image (28x28) into a 1D vector of 784 numbers.
       This throws away spatial structure (where pixels are relative to
       each other), but the network can still learn patterns.
    2. Pass through two hidden layers with ReLU activation.
       Each neuron in a layer is connected to EVERY neuron in the next layer.
    3. Output layer produces 10 scores (one per digit).

    --- Why call it "MLP"? ---
    - "Multi-Layer": has 3 layers (input -> hidden -> hidden -> output).
    - "Perceptron": a single neuron (old-school term).
    - "Fully Connected": every neuron connects to every neuron in next layer.

    --- Pros/Cons ---
    + Simple to understand and implement.
    + Fast to train.
    - Doesn't use spatial information (pixel positions matter a lot for images).
    - ~97% accuracy on MNIST (decent but not state-of-the-art).

    --- Parameter count ---
    Layer 1: 784 inputs × 256 neurons + 256 biases = 200,960
    Layer 2: 256 × 256 + 256 = 65,792
    Layer 3: 256 × 10 + 10 = 2,570
    Total: ~269K parameters
    """

    def __init__(self, hidden_dim: int = 256):
        """
        Args:
            hidden_dim: Number of neurons in the hidden layers.
                        More neurons = more capacity to learn, but slower.
        """
        super().__init__()
        # nn.Flatten converts (B, 1, 28, 28) -> (B, 784).
        self.flatten = nn.Flatten()

        # nn.Linear(in_features, out_features) = fully connected layer.
        # Internally stores:
        #   - weight: (out_features, in_features) tensor
        #   - bias:   (out_features,) tensor
        # Computation: output = input @ weight.T + bias
        self.fc1 = nn.Linear(784, hidden_dim)        # 784 -> 256
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 256 -> 256
        self.out = nn.Linear(hidden_dim, 10)         # 256 -> 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: defines how input transforms to output.

        x shape: (batch_size, 1, 28, 28)
        return shape: (batch_size, 10)
        """
        x = self.flatten(x)        # (B, 1, 28, 28) -> (B, 784)
        x = F.relu(self.fc1(x))    # (B, 784) -> (B, 256), then apply ReLU
        x = F.relu(self.fc2(x))    # (B, 256) -> (B, 256), then apply ReLU
        return self.out(x)          # (B, 256) -> (B, 10)  (raw scores / logits)


class CNN(nn.Module):
    """
    Convolutional Neural Network.

    --- How it works ---
    Unlike MLP which flattens the image immediately, CNN uses convolutional
    layers that slide small "filters" (kernels) across the image.

    Key insight: Convolutions PRESERVE spatial structure.
    A 3x3 filter slides over the image and looks at small 3x3 patches.
    This lets the network learn things like:
      - Layer 1: edges (horizontal, vertical, diagonal lines).
      - Layer 2: shapes (corners, curves, circles).
      - Dense layers: high-level features (the digit "3" has two curves, etc.).

    --- Architecture ---
    Input: (B, 1, 28, 28)
      -> Conv1 (1->32 channels, 3x3 kernel)   -> (B, 32, 26, 26)
      -> ReLU + MaxPool2d(2)                   -> (B, 32, 13, 13)
      -> Conv2 (32->64 channels, 3x3 kernel)   -> (B, 64, 11, 11)
      -> ReLU + MaxPool2d(2)                   -> (B, 64, 5, 5)
      -> Dropout (regularization)
      -> Flatten                               -> (B, 64*5*5) = (B, 1600)
      -> FC1 (1600 -> 128) + ReLU + Dropout
      -> Output (128 -> 10)

    Wait - the shapes above (5x5 at the end) don't match the code (12x12)!
    Let me trace the actual shapes step by step:

        Input:        (B, 1, 28, 28)
        Conv1 (3x3):  (B, 32, 26, 26)   [28 - 3 + 1 = 26]
        MaxPool (2):  (B, 32, 13, 13)   [26 / 2 = 13]
        Conv2 (3x3):  (B, 64, 11, 11)   [13 - 3 + 1 = 11]
        MaxPool (2):  (B, 64, 5, 5)     [11 / 2 = 5]
        Flatten:      (B, 64*5*5) = (B, 1600)

    Hmm, but the code uses 64 * 12 * 12 = 64 * 144 = 9216 for the Linear layer.
    That's because the conv layers use padding or different kernel sizes.
    Actually looking at the code more carefully:

    The code has:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # no padding, so 28->26
        self.pool = nn.MaxPool2d(2)                     # 26->13
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13->11

    But wait, there are TWO pool layers applied (once after conv1, once after conv2):
        28 -> conv1(3) -> 26 -> pool(2) -> 13
        13 -> conv2(3) -> 11 -> pool(2) -> 5

    Hmm, but the code does:
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 26 -> 13
        x = self.pool(F.relu(self.conv2(x)))  # 13 -> 11 -> 5

    So the flattened size should be 64 * 5 * 5 = 1600, not 64 * 12 * 12 = 9216.

    But wait - I wrote the code and I see that the code is what I wrote. Let me
    check the exact numbers. Actually, there IS a mistake here. Let me trace:

    Input: (B, 1, 28, 28)
    Conv1(1,32,3): (B, 32, 26, 26)   because no padding
    Pool(2):       (B, 32, 13, 13)   because floor(26/2) = 13

    Conv2(32,64,3): (B, 64, 11, 11)  because 13-3+1 = 11
    Pool(2):        (B, 64, 5, 5)    because floor(11/2) = 5

    Flatten: (B, 64*5*5) = (B, 1600)

    So 64 * 12 * 12 = 9216 is WRONG in the code. But wait, I wrote this code.
    Let me check... The code says:
        self.fc1 = nn.Linear(64 * 12 * 12, 128)

    64*12*12 = 9216. But the actual size is 64*5*5 = 1600.

    This means if I run this forward pass, it will crash because the Flatten
    layer produces 1600 features but fc1 expects 9216.

    So let me fix this. The correct value should be 64 * 5 * 5 = 1600.
    Actually wait - maybe I should use a different architecture where the
    final spatial size IS 12x12. Let me think...

    If I want 12x12 at the end, I could:
    1. Use padding=1 in conv layers: 28->28, then pool->14. Then 14->14, pool->7. No.
    2. Use only ONE pooling layer.

    Let me think of a standard simple CNN for MNIST. A common one is:
    conv1(1,32,3) -> relu -> conv2(32,64,3) -> relu -> pool(2) -> dropout -> flatten -> fc1 -> dropout -> out

    With this, shapes are:
    28 -> conv1(3) -> 26 -> conv2(3) -> 24 -> pool(2) -> 12

    Yes! 12x12 at the end. Let me fix the forward pass to match.

    So the architecture should be:
    x = F.relu(self.conv1(x))     # 28 -> 26
    x = F.relu(self.conv2(x))     # 26 -> 24
    x = self.pool(x)              # 24 -> 12
    x = self.dropout1(x)
    x = torch.flatten(x, 1)       # (B, 64*12*12) = (B, 9216)
    x = F.relu(self.fc1(x))       # (B, 128)
    x = self.dropout2(x)
    return self.out(x)            # (B, 10)

    That's the intended architecture. Let me fix this.

    --- Parameter count ---
    Conv1:   1*32*3*3 + 32 = 320
    Conv2:   32*64*3*3 + 64 = 18,496
    FC1:     9216*128 + 128 = 1,179,776
    Out:     128*10 + 10 = 1,290
    Total:   ~1.2M parameters
    """

    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        #
        # What is a "channel"?
        #   - Grayscale images = 1 channel (intensity).
        #   - RGB images = 3 channels (red, green, blue).
        #   - After Conv1: we create 32 "feature maps" (learned filters).
        #     Each filter detects a different pattern (horizontal edge,
        #     vertical edge, etc.).
        #
        # What is kernel_size?
        #   - 3 means 3x3 pixel filter.
        #   - Each output pixel "sees" a 3x3 region of the input.
        #   - The filter slides with stride 1 (default), so output is
        #     slightly smaller: 28 -> 26.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 1 -> 32 channels, 28->26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 32 -> 64 channels, 26->24

        # MaxPool2d(2) reduces spatial size by half.
        # Takes the maximum value in each 2x2 block.
        # Why? Reduces computation, makes features more robust to small shifts.
        # 24 -> 12.
        self.pool = nn.MaxPool2d(2)

        # Dropout: randomly "drops" (sets to zero) a fraction of neurons
        # during training. This prevents the network from relying too much
        # on any single neuron (regularization).
        # dropout1 = 25% chance to zero out a neuron.
        # dropout2 = 50% chance to zero out before the final layer.
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # After conv layers, the spatial size is 12x12 with 64 channels.
        # flatten -> 64 * 12 * 12 = 9216 features.
        # FC layer compresses 9216 -> 128.
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.out = nn.Linear(128, 10)                   # 128 -> 10 digits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, 28, 28)
        Output: (B, 10)

        Shape trace through the network:
          (B,  1, 28, 28)  input
          (B, 32, 26, 26)  after conv1 (kernel=3, no padding)
          (B, 32, 24, 24)  after conv2 (kernel=3, no padding)
          (B, 32, 12, 12)  after pool  (2x2)
          (B, 64, 12, 12)  through both convs + pool, then...
          Actually let's trace more carefully:
          conv1: (B, 1, 28, 28) -> (B, 32, 26, 26)
          conv2: (B, 32, 26, 26) -> (B, 64, 24, 24)
          pool:  (B, 64, 24, 24) -> (B, 64, 12, 12)
          flatten: (B, 64*12*12) = (B, 9216)
          fc1:     (B, 9216) -> (B, 128)
          out:     (B, 128) -> (B, 10)
        """
        x = F.relu(self.conv1(x))    # (B, 1, 28, 28) -> (B, 32, 26, 26)
        x = F.relu(self.conv2(x))    # (B, 32, 26, 26) -> (B, 64, 24, 24)
        x = self.pool(x)             # (B, 64, 24, 24) -> (B, 64, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)      # (B, 64, 12, 12) -> (B, 9216)
        x = F.relu(self.fc1(x))      # (B, 9216) -> (B, 128)
        x = self.dropout2(x)
        return self.out(x)           # (B, 128) -> (B, 10)


class ViT(nn.Module):
    """
    Vision Transformer (ViT) - applying Transformer attention to images.

    --- Core idea ---
    "An image is worth 16x16 words" — treat image patches like words in a sentence.

    Steps:
      1. Split the 28x28 image into small patches (e.g., 16 patches of 7x7).
      2. Flatten each patch and project to a vector ("patch embedding").
      3. Add a special [CLS] token (like BERT) whose output will be used for classification.
      4. Add position embeddings so the model knows where each patch came from.
      5. Feed through Transformer encoder layers (self-attention + MLP).
      6. Take the [CLS] token's output and classify it with a linear layer.

    --- Why is this interesting? ---
    Transformers are the foundation of GPT, BERT, Llama, etc.
    ViT shows that the same architecture works for images too!
    Understanding this code transfers directly to understanding LLMs.

    --- Pros/Cons ---
    + Introduces attention, which is the dominant architecture in ML today.
    + More parameter-efficient at large scales.
    - Needs more data than CNN for small datasets (MNIST is tiny).
    - Slower to train on small images (CNNs are more efficient here).
    """

    def __init__(self, patch_size: int = 7, dim: int = 64, depth: int = 4, heads: int = 4):
        """
        Args:
            patch_size: Height/width of each patch (7x7 gives 16 patches for a 28x28 image).
            dim:        Embedding dimension (size of each token vector).
            depth:      Number of Transformer encoder layers.
            heads:      Number of attention heads (multi-head attention).
        """
        super().__init__()

        # Calculate dimensions from the inputs.
        # 28x28 image with 7x7 patches = 16 patches (4x4 grid).
        num_patches = (28 // patch_size) ** 2  # (28/7)^2 = 4^2 = 16

        # Each patch is patch_size * patch_size * 1 channel = 7*7*1 = 49 pixels.
        # We project these 49 pixels into a `dim`-dimensional vector.
        patch_dim = patch_size * patch_size * 1

        self.patch_size = patch_size

        # Linear projection: 49 pixels -> dim=64 dimensional embedding.
        # This is like an embedding layer for image patches.
        self.patch_proj = nn.Linear(patch_dim, dim)

        # Positional embedding: tells the model where each patch belongs.
        # Shape: (1, num_patches + 1, dim)
        #   - +1 for the [CLS] token.
        #   - 1 in the first dim means the same embedding is used for all
        #     images in a batch (broadcast).
        #   - These are LEARNED parameters (the model figures out the
        #     best position representations during training).
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # [CLS] token: a special learnable vector prepended to the sequence.
        #   - Analogous to BERT's [CLS] token.
        #   - At the output, the [CLS] token's representation is used for
        #     classification (not the patch tokens).
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer Encoder layer.
        #   - nhead=heads: multi-head self-attention with `heads` parallel heads.
        #   - batch_first=True: input shape is (batch, seq_len, dim).
        #   - Internally: LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> MLP -> Residual
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True
        )
        # Stack multiple encoder layers to create a deeper Transformer.
        # depth=4 means 4 layers of self-attention + MLP.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final layer norm before classification head.
        self.norm = nn.LayerNorm(dim)

        # Classification head: dim -> 10 classes.
        self.out = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, 28, 28)
        Output: (B, 10)
        """
        B = x.shape[0]  # Batch size

        # --- Step 1: Extract patches ---
        # x.unfold(dim, size, step) "unfolds" a tensor along a dimension.
        # We unfold height (dim=2) and width (dim=3) with size=patch_size, step=patch_size.
        # Each 7x7 block becomes a separate "patch".
        # After unfolding:
        #   shape: (B, 1, 4, 4, 7, 7)   (4 patches in each direction)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        # --- Step 2: Flatten each patch ---
        # Group all patches into a sequence dimension.
        # contiguous() ensures memory is contiguous (required for view()).
        # view(B, num_patches, patch_pixels)
        #   shape: (B, 16, 49)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)

        # --- Step 3: Project patches to embeddings ---
        # patch_proj: 49 -> dim (64)
        #   shape: (B, 16, 64)
        tokens = self.patch_proj(patches)

        # --- Step 4: Prepend [CLS] token ---
        # Expand [CLS] token from (1, 1, 64) -> (B, 1, 64) for each batch.
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate: [CLS] + all patch tokens.
        #   shape: (B, 17, 64)  (1 CLS + 16 patches)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # --- Step 5: Add position embeddings ---
        # Tells the model the spatial arrangement of patches.
        #   shape: (B, 17, 64)
        tokens = tokens + self.pos_embed

        # --- Step 6: Pass through Transformer ---
        # Self-attention lets every token "look at" every other token.
        # After 4 layers, each token has rich contextual information.
        #   shape: (B, 17, 64)
        tokens = self.transformer(tokens)

        # --- Step 7: Classify using [CLS] token ---
        # Take only the [CLS] token (index 0) from the output sequence.
        # Apply LayerNorm before the classification head.
        #   shape: (B, 64)
        cls_out = self.norm(tokens[:, 0])

        # Final linear projection to 10 classes.
        #   shape: (B, 10)
        return self.out(cls_out)


def build_model(name: str) -> nn.Module:
    """
    Factory function: returns a model instance by name.

    Usage:
        model = build_model("cnn")   # Returns a CNN() instance
        model = build_model("mlp")   # Returns an MLP() instance
        model = build_model("vit")   # Returns a ViT() instance

    Args:
        name: One of "mlp", "cnn", "vit" (case-insensitive).

    Returns:
        An nn.Module ready for training.

    Raises:
        ValueError: If the name is not recognized.
    """
    name = name.lower()
    if name == "mlp":
        return MLP()
    elif name == "cnn":
        return CNN()
    elif name == "vit":
        return ViT()
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: mlp, cnn, vit")
