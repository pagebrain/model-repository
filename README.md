# Model Repository

```bash
export MODEL_NAME="dreamshaper-v7"
export MODEL_URL="https://civitai.com/api/download/models/109123?type=Model&format=SafeTensor&size=pruned&fp=fp16"
```

# Publish to HuggingFace
```bash
export GIT_NAME="pagebrain"
export GIT_EMAIL=""

git config --global credential.helper store
git config --global user.name $GIT_NAME
git config --global user.email $GIT_EMAIL

pip install huggingface_hub diffusers==0.17.1 transformers==4.30.2 omegaconf==2.3.0 accelerate==0.22.0 cog==0.8.6
sudo apt-get install git-lfs
git lfs install


wget -O model.safetensors $MODEL_URL
wget -O config.yaml "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
wget -O convert.py https://raw.githubusercontent.com/huggingface/diffusers/v0.17.1/scripts/convert_original_stable_diffusion_to_diffusers.py

huggingface-cli login
huggingface-cli repo create $MODEL_NAME --type model
git clone https://huggingface.co/pagebrain/$MODEL_NAME $MODEL_NAME

python convert.py --checkpoint_path model.safetensors --from_safetensors --to_safetensors --original_config config.yaml --dump_path $MODEL_NAME --half

cd $MODEL_NAME
git add .
git commit -m "Add Diffusers weights for $MODEL_NAME"
git push origin main
```

# Publish to Replicate
```bash
mkdir $MODEL_NAME
mkdir $MODEL_NAME/script
envsubst < predict.py > $MODEL_NAME/predict.py
envsubst < script/download-weights > $MODEL_NAME/script/download-weights
sudo chmod +x $MODEL_NAME/script/download-weights
envsubst < cog.yaml > $MODEL_NAME/cog.yaml

cog login
cd $MODEL_NAME
cog run script/download-weights
cog predict -i prompt="monkey scuba diving"
cog push
```


| CivitAI | HuggingFace | Replicate |
|---|---|---|
| [epiCRealism v2](https://civitai.com/models/25694?modelVersionId=101593) | [pagebrain/epicrealism-v2](https://huggingface.co/pagebrain/epicrealism-v2) | [pagebrain/epicrealism-v2](https://replicate.com/pagebrain/epicrealism-v2) |
| [epiCPhotoGasm v1](https://civitai.com/models/132632/epicphotogasm) | [pagebrain/epicphotogasm-v1](https://huggingface.co/pagebrain/epicphotogasm-v1) | [pagebrain/epicphotogasm-v1](https://replicate.com/pagebrain/epicphotogasm-v1) |
|  |  |  |
|  |  |  |
