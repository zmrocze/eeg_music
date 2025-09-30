"""
Utility functions for freezing/unfreezing specific parts of the EEG-to-music model.
"""

from .eegpt import EegptLightning
import torch
from typing import Optional, Tuple, Dict, Any


def freeze_all_except_head_and_adapters(
  model: EegptLightning, verbose: bool = True
) -> None:
  """
  Freeze all model parameters except:
  1. chan_conv - channel convolution layer
  2. head - the final linear classification layer in EEGPTClassifier
  3. linear - the ResidualLinear module (both linear1 and linear2)

  This allows fine-tuning only the task-specific adaptation layers while
  keeping the pretrained encoder and reconstructor/predictor frozen.

  Args:
      model: The EegptLightning model
      verbose: If True, print information about trainable/frozen parameters

  Model structure:
      EegptLightning
      └── model (EegptWithLinear)
          ├── model (EEGPTClassifier)
          │   ├── chan_conv - TRAINABLE
          │   ├── target_encoder (EEGTransformer) - FROZEN
          │   ├── reconstructor/predictor - FROZEN
          │   └── head (LinearWithConstraint) - TRAINABLE
          └── linear (ResidualLinear) - TRAINABLE
              ├── linear1 - TRAINABLE
              └── linear2 - TRAINABLE

  Note: Will raise AttributeError if expected attributes don't exist.
  """

  # Extract the nested model structure from EegptLightning
  eegpt_with_linear = model.model
  eegpt_classifier = model.model.model

  # First, freeze everything
  for param in model.parameters():
    param.requires_grad = False

  # Unfreeze chan_conv - will error if it doesn't exist
  for param in eegpt_classifier.chan_conv.parameters():
    param.requires_grad = True

  # Unfreeze the head (final classification layer in EEGPTClassifier)
  for param in eegpt_classifier.head.parameters():
    param.requires_grad = True

  # Unfreeze the ResidualLinear module
  for param in eegpt_with_linear.linear.parameters():
    param.requires_grad = True

  if verbose:
    print("\n" + "=" * 80)
    print("TRAINABLE PARAMETERS:")
    print("=" * 80)

    trainable_params = 0
    frozen_params = 0

    # Count and display trainable parameters by component
    components: Dict[str, Optional[Tuple[int, int]]] = {
      "chan_conv": None,
      "head": None,
      "linear (ResidualLinear)": None,
      "target_encoder": None,
      "reconstructor/predictor": None,
    }

    # chan_conv
    chan_conv_module = eegpt_classifier.chan_conv
    chan_conv_params = sum(p.numel() for p in chan_conv_module.parameters())
    chan_conv_trainable = sum(
      p.numel() for p in chan_conv_module.parameters() if p.requires_grad
    )
    components["chan_conv"] = (chan_conv_trainable, chan_conv_params)
    trainable_params += chan_conv_trainable
    frozen_params += chan_conv_params - chan_conv_trainable

    # head
    head_module = eegpt_classifier.head
    head_params = sum(p.numel() for p in head_module.parameters())
    head_trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)
    components["head"] = (head_trainable, head_params)
    trainable_params += head_trainable
    frozen_params += head_params - head_trainable

    # linear (ResidualLinear)
    linear_module = eegpt_with_linear.linear
    linear_params = sum(p.numel() for p in linear_module.parameters())
    linear_trainable = sum(
      p.numel() for p in linear_module.parameters() if p.requires_grad
    )
    components["linear (ResidualLinear)"] = (linear_trainable, linear_params)
    trainable_params += linear_trainable
    frozen_params += linear_params - linear_trainable

    # target_encoder
    encoder_module = eegpt_classifier.target_encoder
    encoder_params = sum(p.numel() for p in encoder_module.parameters())
    encoder_trainable = sum(
      p.numel() for p in encoder_module.parameters() if p.requires_grad
    )
    components["target_encoder"] = (encoder_trainable, encoder_params)
    trainable_params += encoder_trainable
    frozen_params += encoder_params - encoder_trainable

    # Check for reconstructor or predictor
    if hasattr(eegpt_classifier, "reconstructor"):
      rec_module = eegpt_classifier.reconstructor
      rec_params = sum(p.numel() for p in rec_module.parameters())
      rec_trainable = sum(p.numel() for p in rec_module.parameters() if p.requires_grad)
      components["reconstructor/predictor"] = (rec_trainable, rec_params)
      trainable_params += rec_trainable
      frozen_params += rec_params - rec_trainable
    elif hasattr(eegpt_classifier, "predictor"):
      pred_module = eegpt_classifier.predictor
      pred_params = sum(p.numel() for p in pred_module.parameters())
      pred_trainable = sum(
        p.numel() for p in pred_module.parameters() if p.requires_grad
      )
      components["reconstructor/predictor"] = (pred_trainable, pred_params)
      trainable_params += pred_trainable
      frozen_params += pred_params - pred_trainable

    # Print component details
    for component_name, counts in components.items():
      if counts is not None:
        trainable, total = counts
        status = "✓ TRAINABLE" if trainable > 0 else "✗ FROZEN"
        print(f"{component_name:30s}: {trainable:>10,} / {total:>10,} params  {status}")

    print("-" * 80)
    print(f"{'TOTAL TRAINABLE':30s}: {trainable_params:>10,} params")
    print(f"{'TOTAL FROZEN':30s}: {frozen_params:>10,} params")
    print(f"{'TOTAL':30s}: {trainable_params + frozen_params:>10,} params")
    print(
      f"{'Trainable ratio':30s}: {100 * trainable_params / (trainable_params + frozen_params):.2f}%"
    )
    print("=" * 80)

    print("\nWHAT IS TRAINABLE:")
    print("  ✓ chan_conv: Channel-wise 1D convolution (adapts input channels to model)")
    print(
      "  ✓ head: Final linear layer in EEGPTClassifier (maps embeddings to num_classes)"
    )
    print("  ✓ linear (ResidualLinear): Two-layer residual network (linear1 + linear2)")
    print("    - linear1: input_dim -> input_dim with residual connection")
    print("    - linear2: input_dim -> output_dim (final projection)")

    print("\nWHAT IS FROZEN:")
    print(
      "  ✗ target_encoder: Pretrained EEG encoder (patch embedding, transformer blocks)"
    )
    print("  ✗ reconstructor/predictor: Pretrained reconstruction/prediction head")

    print("\nNOTE: You might also want to make trainable:")
    print("  - norm/fc_norm: Layer normalization before the head")
    print("  - chan_embed: Channel embedding layer in target_encoder")
    print("  - cls_token/summary_token: Learnable tokens")
    print("=" * 80 + "\n")


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
  """
  Count trainable and total parameters in the model.

  Args:
      model: PyTorch model

  Returns:
      Dictionary with 'trainable', 'frozen', and 'total' parameter counts
  """
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  frozen = total - trainable

  return {
    "trainable": trainable,
    "frozen": frozen,
    "total": total,
    "trainable_ratio": trainable / total if total > 0 else 0,
  }
