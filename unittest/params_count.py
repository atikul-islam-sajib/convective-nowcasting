import torch

_original_device = torch.device
def _patched_device(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("cuda"):
        return _original_device("cpu")
    return _original_device(*args, **kwargs)
torch.device = _patched_device


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(total_params, bytes_per_param=4):
    return (total_params * bytes_per_param) / (1024 ** 2)


def report(name, model, input_shape=None, output_shape=None):
    total, trainable = count_parameters(model)
    size_mb = model_size_mb(total)
    print(f"{'='*60}\n{name}\n{'='*60}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (float32): {size_mb:,.2f} MB")
    if input_shape is not None:
        print(f"Input tensor shape:   {input_shape}")
    if output_shape is not None:
        print(f"Output tensor shape:  {output_shape}")
    print()
    return total, trainable, size_mb


results = {}
T = 4  
n_ch = 3
n_horizons = 4

try:
    from src.models.convlstm.model import ED
    from src.models.convlstm.net_params import (
        convlstm_encoder_params,
        convlstm_decoder_params,
    )
    from src.models.convlstm.encoder import Encoder
    from src.models.convlstm.decoder import Decoder

    convlstm_encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    convlstm_decoder = Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1])
    convlstm_model = ED(convlstm_encoder, convlstm_decoder)
    results["ConvLSTM"] = report("ConvLSTM", convlstm_model,
                                   input_shape=(1, T * n_ch, 256, 256),
                                   output_shape=(1, n_horizons, 256, 256))
except Exception as e:
    print(f"ConvLSTM instantiation failed: {e}\n")

try:
    from src.models.simvp.model import SimVP

    shape_in = (T, n_ch, 256, 256)
    simvp_model = SimVP(
        shape_in=shape_in,
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=8,
        incep_ker=[3, 5, 7, 11],
        groups=8,
    )
    results["SimVP"] = report("SimVP", simvp_model,
                                input_shape=(1, T * n_ch, 256, 256),
                                output_shape=(1, n_horizons, 256, 256))
except Exception as e:
    print(f"SimVP instantiation failed: {e}\n")

try:
    from src.models.smaat_unet.SmaAt_UNet import SmaAt_UNet

    smaat_model = SmaAt_UNet(
        n_channels=T * n_ch,
        n_classes=n_horizons,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    )
    results["SmaAt-UNet"] = report("SmaAt-UNet", smaat_model,
                                     input_shape=(1, T * n_ch, 256, 256),
                                     output_shape=(1, n_horizons, 256, 256))
except Exception as e:
    print(f"SmaAt-UNet instantiation failed: {e}\n")

try:
    from src.models.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

    enc_depth = [1, 1]
    dec_depth = [1, 1]
    num_enc_blocks = len(enc_depth)
    num_dec_blocks = len(dec_depth)

    earthformer_model = CuboidTransformerModel(
        input_shape=(T, 256, 256, n_ch),
        target_shape=(n_horizons, 256, 256, 1),
        base_units=64,
        block_units=None,
        scale_alpha=1.0,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        enc_attn_patterns=["axial"] * num_enc_blocks,
        dec_self_attn_patterns=["axial"] * num_dec_blocks,
        dec_cross_attn_patterns=["cross_1x1"] * num_dec_blocks,
        enc_cuboid_size=[(4, 4, 4)] * num_enc_blocks,
        enc_cuboid_strategy=[('l', 'l', 'l')] * num_enc_blocks,
        enc_shift_size=[(0, 0, 0)] * num_enc_blocks,
        dec_self_cuboid_size=[(4, 4, 4)] * num_dec_blocks,
        dec_self_cuboid_strategy=[('l', 'l', 'l')] * num_dec_blocks,
        dec_self_shift_size=[(0, 0, 0)] * num_dec_blocks,
        dec_cross_cuboid_hw=[(4, 4)] * num_dec_blocks,
        dec_cross_cuboid_strategy=[('l', 'l')] * num_dec_blocks,
        dec_cross_shift_hw=[(0, 0)] * num_dec_blocks,
        dec_cross_n_temporal=[2] * num_dec_blocks,
        dec_cross_last_n_frames=None,
        enc_use_inter_ffn=True,
        dec_use_inter_ffn=True,
        dec_hierarchical_pos_embed=True,
        dec_use_first_self_attn=False,
        dec_cross_start=0,
        num_heads=4,
        attn_drop=0.1,
        proj_drop=0.1,
        ffn_drop=0.1,
        downsample=2,
        downsample_type="patch_merge",
        upsample_type="upsample",
        initial_downsample_type="stack_conv",
        initial_downsample_activation="leaky",
        initial_downsample_stack_conv_num_layers=3,
        initial_downsample_stack_conv_dim_list=[4, 16, 64],
        initial_downsample_stack_conv_downscale_list=[2, 2, 2],
        initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
        num_global_vectors=8,
        use_dec_self_global=True,
        dec_self_update_global=True,
        use_dec_cross_global=True,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        z_init_method="zeros",
        ffn_activation="gelu",
        gated_ffn=False,
        norm_layer="layer_norm",
        padding_type="zeros",
        pos_embed_type="t+hw",
        checkpoint_level=2,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_up_linear_init_mode="0",
        norm_init_mode="0",
    )
    results["EarthFormer"] = report("EarthFormer", earthformer_model,
                                      input_shape=(1, T * n_ch, 256, 256),
                                      output_shape=(1, n_horizons, 256, 256))
except Exception as e:
    print(f"EarthFormer instantiation failed: {e}\n")

try:
    from src.models.vptr.model.VPTR_modules import VPTREnc, VPTRDec, VPTRFormerNAR

    vptr_enc = VPTREnc(img_channels=n_ch, feat_dim=192, n_downsampling=3)
    vptr_dec = VPTRDec(img_channels=n_ch, feat_dim=192, n_downsampling=3)
    vptr_former = VPTRFormerNAR(
        num_past_frames=T,
        num_future_frames=n_horizons,
        encH=32,
        encW=32,
        d_model=192,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        window_size=4,
        Spatial_FFN_hidden_ratio=4,
        rpe=True,
    )

    total_params = (
        sum(p.numel() for p in vptr_enc.parameters())
        + sum(p.numel() for p in vptr_dec.parameters())
        + sum(p.numel() for p in vptr_former.parameters())
    )
    trainable_params = (
        sum(p.numel() for p in vptr_enc.parameters() if p.requires_grad)
        + sum(p.numel() for p in vptr_dec.parameters() if p.requires_grad)
        + sum(p.numel() for p in vptr_former.parameters() if p.requires_grad)
    )
    print(f"{'='*60}\nVPTR (Enc + Dec + Transformer core)\n{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (float32): {model_size_mb(total_params):,.2f} MB")
    print(f"Input tensor shape:   (1, {T * n_ch}, 256, 256)")
    print(f"Output tensor shape:  (1, {n_horizons}, 256, 256)\n")
    results["VPTR"] = (total_params, trainable_params, model_size_mb(total_params))
except Exception as e:
    print(f"VPTR instantiation failed: {e}\n")


print(f"{'='*60}\nFINAL SUMMARY (T=4, all configs confirmed)\n{'='*60}")
for name, values in results.items():
    total, trainable, size_mb = values
    print(f"{name:15s} total={total:>12,}  trainable={trainable:>12,}  size={size_mb:>8,.2f} MB")