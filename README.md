# vit-cnn-self-supervised-architecture
This project aims to compare different image classification models, including Convolutional Neural Networks (CNN), Vision Transformers (ViT), and Vision Transformers with Masked Auto Encoding (MAE). The objective is to analyze the performance of these models in terms of accuracy, loss, and their ability to generalize on unseen data.

## Models
CNN: A deep learning architecture based on convolutional layers specifically designed for image classification tasks.

CNN with SimCLR: A CNN model augmented with the SimCLR contrastive learning method to improve feature learning and generalization.

ViT: A transformer-based model that uses self-attention mechanisms to process image data, achieving state-of-the-art performance on several image classification benchmarks.

ViT with MAE: A ViT model that incorporates masked autoencoding to learn more expressive features and improve classification performance.

## Results
Here is a summary of the results obtained from each model on a brain tumor MRI dataset:

Test Accuracy:

CNN: 74.37%	

CNN with SimCLR: 76.14%

ViT:	79.16%	

ViT with MAE: 83.11%

## Analysis
ViT and ViT with MAE outperform the CNN models in terms of test accuracy.
Adding SimCLR to the CNN model results in a significant improvement in performance.
ViT with MAE achieves the highest training accuracy, indicating effective learning of training data. However, overfitting should be considered as a potential issue.
The validation accuracies for both ViT and ViT with MAE are higher than the corresponding test accuracies, suggesting good generalization on the validation set but possible discrepancies between the validation and test sets.


## Future Work
Investigate the possibility of overfitting for models with high training accuracy.
Perform qualitative analysis on a few random patients to gain insights into the model's performance and behavior.
Explore additional augmentation techniques and regularization methods to further improve model performance.
Analyze the impact of different hyperparameters on the performance of each model.

## Acknowledgments
This project was developed using open-source libraries such as PyTorch, torchvision, and timm. We appreciate the contributions of the developers and maintainers of these libraries.

Data source:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
