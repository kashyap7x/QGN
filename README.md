# QuadNet for Semantic Segmentation on PyTorch

This is a PyTorch implementation of QuadNet, a hierarchical scene parsing network based on graph convolutions. 

## Issues
* ResNet-34 doesn't work with synchronized batch norm
* Dataloader returning a single item list instead of dict as feed_dict for single GPU
* No .cuda() in the original code
* Check utils.to_one_hot and KLDivLoss() 

## References
[Codebase Repository](https://github.com/CSAILVision/semantic-segmentation-pytorch)
