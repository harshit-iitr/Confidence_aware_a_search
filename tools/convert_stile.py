import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.torch_model import ChrestienHeuristicNet
from tools.convert_finalSok3 import read_full_tf_index, parse_bundle_entry


def convert_stile():
    index_path = "Optimize-Planning-Heuristics-to-Rank/slidingtile/train/stile.index"
    data_path = "Optimize-Planning-Heuristics-to-Rank/slidingtile/train/stile.data-00000-of-00001"

    kvs = read_full_tf_index(index_path)
    weights = {}

    with open(data_path, "rb") as f_data:
        for k, entry_bytes in kvs.items():
            if ".ATTRIBUTES/VARIABLE_VALUE" in k and "optimizer" not in k.lower():
                entry = parse_bundle_entry(entry_bytes)
                if 'size' in entry and entry['size'] > 0:
                    offset = entry.get('offset', 0)
                    f_data.seek(offset)
                    raw_bytes = f_data.read(entry['size'])
                    arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                    if 'shape' in entry and entry['shape'] and np.prod(entry['shape']) == arr.size:
                        arr = arr.reshape(entry['shape'])
                    weights[k] = arr

    print(f"Decoded {len(weights)} active weight tensors from stile checkpoint.")

    model = ChrestienHeuristicNet(dim=5, in_channels=50)
    sd = model.state_dict()

    def tf_conv_to_pt(arr):
        return torch.from_numpy(arr).permute(3, 2, 0, 1)

    def tf_dense_to_pt(arr):
        return torch.from_numpy(arr).t()

    mapping = [
        ("layer_with_weights-0", "conv1", "conv"),   # conv1 (3, 3, 50, 64)
        ("layer_with_weights-1", "conv2", "conv"),   # conv2 (3, 3, 114, 64)
        ("layer_with_weights-2", "conv3", "conv"),   # conv3 (3, 3, 114, 64)
        ("layer_with_weights-3", "conv4", "conv"),   # conv4 (3, 3, 114, 64)
        ("layer_with_weights-4", "conv5", "conv"),   # conv5 (3, 3, 114, 64)
        ("layer_with_weights-5", "conv6", "conv"),   # conv6 (3, 3, 114, 64)
        ("layer_with_weights-6", "conv7", "conv"),   # conv7 (3, 3, 114, 64)
        ("layer_with_weights-8", "conv8", "conv"),   # conv8-2 (3, 3, 114, 180)
        ("layer_with_weights-10", "conv9", "conv"),  # conv9-2 (3, 3, 290, 180)
        ("layer_with_weights-12", "conv10", "conv"), # conv10-2 (3, 3, 290, 180)
        ("layer_with_weights-14", "conv11", "conv"), # conv11-2 (3, 3, 290, 180)
        ("layer_with_weights-16", "fc1", "dense"),   # dense-2 (290, 256)
        ("layer_with_weights-18", "fc2", "dense"),   # op-2 (256, 1)
    ]

    for tf_prefix, pt_name, ltype in mapping:
        k_key = f"{tf_prefix}/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        b_key = f"{tf_prefix}/bias/.ATTRIBUTES/VARIABLE_VALUE"

        if k_key in weights:
            if ltype == "conv":
                sd[f"{pt_name}.weight"] = tf_conv_to_pt(weights[k_key])
            else:
                sd[f"{pt_name}.weight"] = tf_dense_to_pt(weights[k_key])
            print(f"Loaded {pt_name}.weight: {sd[f'{pt_name}.weight'].shape}")
        if b_key in weights:
            sd[f"{pt_name}.bias"] = torch.from_numpy(weights[b_key])
            print(f"Loaded {pt_name}.bias: {sd[f'{pt_name}.bias'].shape}")

    # In the authors' TensorFlow AttentionAugmentation2D, inputs (180 channels from conv8..11)
    # are split directly into [q(60), k(60), v(60)] with no extra 1x1 qkv_conv layer.
    # Setting qkv_conv to Identity makes PyTorch AttentionAugmentation2D an exact 1:1 replica.
    eye_180 = torch.eye(180, dtype=torch.float32).view(180, 180, 1, 1)
    zero_180 = torch.zeros(180, dtype=torch.float32)
    for att_idx in range(1, 5):
        sd[f"att{att_idx}.qkv_conv.weight"] = eye_180.clone()
        sd[f"att{att_idx}.qkv_conv.bias"] = zero_180.clone()
    print("Set att1..att4.qkv_conv to exact 180x180 Identity projection (matching TF twodattention.py).")

    model.load_state_dict(sd, strict=True)
    os.makedirs("checkpoints/slidingtile", exist_ok=True)
    out_path = "checkpoints/slidingtile/stile_paper_pytorch.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved paper's exact pre-trained Sliding-Tile weights to: {out_path}")


if __name__ == "__main__":
    convert_stile()
