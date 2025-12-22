# Semi-Supervised Lesion Classification
***Colaboration with [UrMBCMRabbont](https://github.com/UrMBCMRabbont)***

## Modernized Implementation Notes
- Backbone: ConvNeXt-Tiny (ImageNet pretrained)
- SSL: FixMatch (weak/strong views with RandAugment/TrivialAugment fallback + Cutout)
- Loss: BCEWithLogitsLoss with pos_weight (optional focal)
- Optimizer: AdamW with cosine decay and warmup
- Evaluation: Acc, AUC, AP, with EMA weights for validation
- Outputs: `results/history.csv` and `results/training_curves.png`

*The Final Project 1 of HKUST ELEC4010N - Artificial Intelligence for Medical Image Analysis*

Implementing semi-supervised binary classification on a dermoscopic lesion dataset by Mean Teacher model and ResNet.

For more high-level details, read the Project 1 part of the [presentation slides](./Presentation.pdf) and the [report](./Report.pdf).

Results:
- Validation accuracy: 80.00%
- Validation AUC: 0.6648

These results are compared with the baseline:
- Validation accuracy: 81.13%
- Validation AUC: 0.5402

Note that this dataset has a class imbalance problem, so it is likely to have high validation accuracy with low or high validation AUC. More tunings might produce better results.

## Prerequisites
Download and unzip data from the following links:
- [ISBI2016_ISIC_Part3_Training_Data.zip](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip)
- [ISBI2016_ISIC_Part3_Training_GroundTruth.csv](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv)

Place the files into the main directory. Alternatively, run the following commands in the notebook:

```python
!wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip
!wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv
!unzip "./ISBI2016_ISIC_Part3_Training_Data.zip"
```

Install the additional library by `pip install -U albumentations`.

These parts are included in the first code cell in the notebook.

Note that the test data are not used in this project. However if need, the test data can be accessed from the links:
- [ISBI2016_ISIC_Part3_Test_Data.zip](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip)
- [ISBI2016_ISIC_Part3_Test_GroundTruth.csv](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv)

## Notebook Outline
0. For Colab
1. Import
2. Data Loading
3. Data Preprocessing and Dataloaders
    1. Upsampling
    2. Augmentation
    3. Dataloaders
4. Building Models
    1. ResNet-50
    2. BCE Focal Loss
    3. Mean Teacher Model
5. Training
6. Results

## Reference
Tarvainen, A., Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets
improve semi-supervised deep learning results. *arXiv:1703.01780*
