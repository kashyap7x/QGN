# QuadNet for Semantic Segmentation on PyTorch

This is a PyTorch implementation of QuadNet, a hierarchical scene parsing network based on graph convolutions. 

## Issues
* ResNet-34 doesn't work with synchronized batch norm
* No .cuda() in the original code
* utils.to_one_hot to be verified

## References
[Original Repository](https://github.com/CSAILVision/semantic-segmentation-pytorch)
