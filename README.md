# comfyui-containers
docker build --platform linux/amd64 -t traysimay/comfyui-flux-redux .
docker push traysimay/comfyui-flux-redux

docker build --platform linux/amd64 -t traysimay/comfyui-ltx .
docker push traysimay/comfyui-ltx

docker build --platform linux/amd64 -t traysimay/comfyui-wan .
docker push traysimay/comfyui-wan


docker build --platform linux/amd64 -t traysimay/comfyui-wan-wrapper-fast .
docker push traysimay/comfyui-wan-wrapper-fast