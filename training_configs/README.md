# Alignment 

    In our experiments, we align the neural representations of N with the symbolic
    variables of C by partitioning the layer resulting from the first application of
    max-pooling into quadrants QTL, QTR, QBL, QBR which are aligned with the variables
    YTL, YTR, YBL, YBR.


Translating this to the mapping aligment:
- Oracle Model
    * "$L:0$[:,:]" -> YTL
    * "$L:1$[:,:]" -> YTR
    * "$L:2$[:,:]" -> YBL
    * "$L:3$[:,:]" -> YBR
- ResNet18 layers are
    * conv1
    * **maxpool**
    * layer1
    * layer2
    * layer3
    * layer4
- Layer resulting from the first max-pooling is L=1
    * Output size torch.Size([1, 64, 20, 20])
- Partioning the layer into quadrants QTL, QTR, QBL, QBR
    * "$L:1$[:10,:10]" -> QTL
    * "$L:1$[:10,10:]" -> QTR
    * "$L:1$[10:,:10]" -> QBL
    * "$L:1$[10:,10:]" -> QBR