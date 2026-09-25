import os
import struct
import numpy as np
import torch
from torch_model import ChrestienHeuristicNet

def decode_varint(data, pos):
    res = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        res |= (b & 0x7f) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, pos

def parse_bundle_entry(entry_bytes):
    pos = 0
    entry = {'offset': 0}  # Protobuf defaults to 0 if omitted
    shape = []
    while pos < len(entry_bytes):
        tag, pos = decode_varint(entry_bytes, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        
        if wire_type == 0:
            val, pos = decode_varint(entry_bytes, pos)
            if field_num == 1:
                entry['dtype'] = val
            elif field_num == 3:
                entry['shard_id'] = val
            elif field_num == 4:
                entry['offset'] = val
            elif field_num == 5:
                entry['size'] = val
        elif wire_type == 2:
            length, pos = decode_varint(entry_bytes, pos)
            sub_data = entry_bytes[pos:pos+length]
            pos += length
            if field_num == 2:
                p_sub = 0
                while p_sub < len(sub_data):
                    t_sub, p_sub = decode_varint(sub_data, p_sub)
                    fn_sub = t_sub >> 3
                    wt_sub = t_sub & 0x07
                    if wt_sub == 2 and fn_sub == 2:
                        l_dim, p_sub = decode_varint(sub_data, p_sub)
                        dim_data = sub_data[p_sub:p_sub+l_dim]
                        p_sub += l_dim
                        p_dim = 0
                        while p_dim < len(dim_data):
                            t_d, p_dim = decode_varint(dim_data, p_dim)
                            if (t_d >> 3) == 1:
                                sz, p_dim = decode_varint(dim_data, p_dim)
                                shape.append(sz)
                            else:
                                if (t_d & 0x07) == 0:
                                    _, p_dim = decode_varint(dim_data, p_dim)
            entry['shape'] = shape
        elif wire_type == 5:
            val = struct.unpack('<I', entry_bytes[pos:pos+4])[0]
            pos += 4
            if field_num == 6:
                entry['crc32c'] = val
        else:
            break
    return entry

def parse_sstable_block(data, pos, limit):
    num_restarts = struct.unpack('<I', data[limit-4:limit])[0]
    restarts_pos = limit - 4 * (num_restarts + 1)
    
    entries = []
    curr_pos = pos
    prev_key = b""
    
    while curr_pos < restarts_pos:
        shared_len, curr_pos = decode_varint(data, curr_pos)
        unshared_len, curr_pos = decode_varint(data, curr_pos)
        val_len, curr_pos = decode_varint(data, curr_pos)
        
        key_delta = data[curr_pos : curr_pos + unshared_len]
        curr_pos += unshared_len
        
        val_bytes = data[curr_pos : curr_pos + val_len]
        curr_pos += val_len
        
        full_key = prev_key[:shared_len] + key_delta
        prev_key = full_key
        entries.append((full_key.decode('latin1'), val_bytes))
        
    return entries

def read_full_tf_index(index_path):
    with open(index_path, "rb") as f:
        data = f.read()
        
    p = len(data) - 48
    _, p = decode_varint(data, p)
    _, p = decode_varint(data, p)
    idx_offset, p = decode_varint(data, p)
    idx_size, p = decode_varint(data, p)
    
    idx_block_entries = parse_sstable_block(data, idx_offset, idx_offset + idx_size)
    
    all_kvs = {}
    for block_key, handle_bytes in idx_block_entries:
        hp = 0
        b_offset, hp = decode_varint(handle_bytes, hp)
        b_size, hp = decode_varint(handle_bytes, hp)
        block_kvs = parse_sstable_block(data, b_offset, b_offset + b_size)
        for k, v in block_kvs:
            all_kvs[k] = v
            
    return all_kvs

def convert():
    index_path = "Optimize-Planning-Heuristics-to-Rank/sokoban/test/finalSok3.index"
    data_path = "Optimize-Planning-Heuristics-to-Rank/sokoban/test/finalSok3.data-00000-of-00001"
    
    kvs = read_full_tf_index(index_path)
    weights = {}
    
    with open(data_path, "rb") as f_data:
        for k, entry_bytes in kvs.items():
            if ".ATTRIBUTES/VARIABLE_VALUE" in k and "OPTIMIZER_SLOT" not in k:
                entry = parse_bundle_entry(entry_bytes)
                if 'size' in entry and entry['size'] > 0:
                    offset = entry.get('offset', 0)
                    f_data.seek(offset)
                    raw_bytes = f_data.read(entry['size'])
                    arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                    if 'shape' in entry and entry['shape'] and np.prod(entry['shape']) == arr.size:
                        arr = arr.reshape(entry['shape'])
                    weights[k] = arr

    print(f"Decoded {len(weights)} active weight tensors from finalSok3.")

    model = ChrestienHeuristicNet(dim=10)
    sd = model.state_dict()
    
    def tf_conv_to_pt(arr):
        return torch.from_numpy(arr).permute(3, 2, 0, 1)

    def tf_dense_to_pt(arr):
        return torch.from_numpy(arr).t()

    mapping = [
        ("layer_with_weights-0", "conv1", "conv"),   # conv1 (3, 3, 10, 64)
        ("layer_with_weights-1", "conv2", "conv"),   # conv2 (3, 3, 74, 64)
        ("layer_with_weights-2", "conv3", "conv"),   # conv3 (3, 3, 74, 64)
        ("layer_with_weights-3", "conv4", "conv"),   # conv4 (3, 3, 74, 64)
        ("layer_with_weights-4", "conv5", "conv"),   # conv5 (3, 3, 74, 64)
        ("layer_with_weights-5", "conv6", "conv"),   # conv6 (3, 3, 74, 64)
        ("layer_with_weights-6", "conv7", "conv"),   # conv7 (3, 3, 74, 64)
        ("layer_with_weights-8", "conv8", "conv"),   # conv8-2 (3, 3, 74, 180)
        ("layer_with_weights-10", "conv9", "conv"),  # conv9-2 (3, 3, 250, 180)
        ("layer_with_weights-12", "conv10", "conv"), # conv10-2 (3, 3, 250, 180)
        ("layer_with_weights-14", "conv11", "conv"), # conv11-2 (3, 3, 250, 180)
        ("layer_with_weights-16", "fc1", "dense"),   # dense-2 (250, 256)
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

    model.load_state_dict(sd, strict=False)
    out_path = "finalSok3_pytorch.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved paper's exact pre-trained weights to: {out_path}")

if __name__ == "__main__":
    convert()
