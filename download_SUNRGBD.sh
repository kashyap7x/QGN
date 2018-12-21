wget -O ./data/sun_rgbd.tgz http://cvgl.stanford.edu/data2/sun_rgbd.tgz
mkdir -p ./data/SUNRGBD
tar -C ./data/SUNRGBD -xzf ./data/sun_rgbd.tgz
rm ./data/sun_rgbd.tgz
echo "Dataset downloaded."
