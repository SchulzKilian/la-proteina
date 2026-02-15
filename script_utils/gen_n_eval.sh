CONFIG_NAME="inference_ucond_notri"

rm -r inference  # This removes all your inference runs so far in that directory
python proteinfoundation/generate.py --config_name inference_ucond_notri \
    ckpt_path=./store/test_release_diffusion/checkpoints \
    ckpt_name=last.ckpt \
    autoencoder_ckpt_path=./checkpoints_laproteina/AE1_ucond_512.ckpt
python proteinfoundation/evaluate.py --config_name $CONFIG_NAME
