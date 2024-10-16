# HGAO

A hybrid algorithm based on Horned Lizard Optimization Algorithm and Giant Armadillo Optimization

## [Appendix](https://github.com/aliwa8168/HGAO/tree/main/Appendix)

Containing one folder and one document.

- [CECTestFunctions-and-Analysis.docx](https://github.com/aliwa8168/HGAO/blob/main/Appendix/CECTestFunctions-and-Analysis.docx):This document contains a detailed process and result analysis of experiments conducted on **CEC2021** and **CEC2022** test functions. In these experiments, the proposed **HGAO** model is comprehensively compared with other state-of-the-art optimization algorithms.

- [Training set accuracy and loss convergence curve](https://github.com/aliwa8168/HGAO/tree/main/Appendix/Training%20set%20accuracy%20and%20loss%20convergence%20curve): This folder contains the accuracy and loss convergence curves for five datasets on the training set. Each dataset includes two images depicting the training set accuracy and two images depicting the training set loss.

  The accuracy convergence curves of HGAO and the other 9 algorithms on the  [LC25000](https://github.com/aliwa8168/HGAO/tree/main/Datasets/LC25000) dataset are shown below.

<p align="center">
  <img src="https://github.com/aliwa8168/HGAO/blob/main/Appendix/Training%20set%20accuracy%20and%20loss%20convergence%20curve/LC25000/Training%20accuracy-1.png" width="45%" />
  <img src="https://github.com/aliwa8168/HGAO/blob/main/Appendix/Training%20set%20accuracy%20and%20loss%20convergence%20curve/LC25000/Training%20accuracy-2.png" width="45%" />
</p>

## [Datasets](https://github.com/aliwa8168/HGAO/tree/main/Datasets)

This folder involves five datasets used in the experiments, four of which are publicly available for download: [LC25000](https://github.com/aliwa8168/HGAO/tree/main/Datasets/LC25000),[UC Merced Land Use Dataset](https://github.com/aliwa8168/HGAO/tree/main/Datasets/UC%20Merced%20Land%20Use%20Dataset), [AIDER](https://github.com/aliwa8168/HGAO/tree/main/Datasets/AIDER), and [PlantVillage](https://github.com/aliwa8168/HGAO/tree/main/Datasets/PlantVillage), with specific download links provided for each. These datasets cover different application scenarios, including medical image classification, remote sensing image analysis, disaster recognition, and plant disease detection. Additionally, a self-made dataset, named the **[CMI5(Chinese Medicine Identification 5 Dataset)](https://github.com/aliwa8168/HGAO/tree/main/Datasets/Chinese%20herbal%20medicine%20Datasets)**, was used for Chinese herbal medicine image classification tasks. This custom dataset consists of images from five categories of Chinese medicinal herbs: mint, fritillaria cirrhosa, honeysuckle, ophiopogon japonicus, and ginseng, with 2020 images for each category. The image resolution is 224x224, totaling 10100 images. 

## [Model_Train](https://github.com/aliwa8168/HGAO/tree/main/Model_Train)

The folder contains the necessary files for running the model, with the specific contents of the files as follows:

- [DenseNet121.py](https://github.com/aliwa8168/HGAO/blob/main/Model_Train/DenseNet121.py): This file contains the detailed implementation of the **DenseNet-121** model architecture.

- **[HGAO.py](https://github.com/aliwa8168/HGAO/blob/main/Model_Train/HGAO.py)**: This is the implementation file for the **HGAO** algorithm, providing the core logic of the HGAO algorithm and the steps for hyperparameter optimization.

- **[model_train.py](https://github.com/aliwa8168/HGAO/blob/main/Model_Train/model_train.py)**: This file is responsible for the entire model training process. It includes the invocation of the HGAO algorithm to find the optimal hyperparameters for DenseNet-121. Additionally, the file implements the specific steps of the model training process, including loading training data, model training and validation, and the final model evaluation.

- **[read_data.py](https://github.com/aliwa8168/HGAO/blob/main/Model_Train/read_data.py)**: This is a file responsible for reading datasets. It contains operations for loading and preprocessing various datasets used in the experiment, including steps like image normalization and resizing. This file ensures that the data is formatted appropriately for model training, providing essential data support for the training and evaluation of DenseNet-121.



## Model training process

After correctly downloading and configuring all datasets, ensure that the runtime environment is properly set up. Next, modify the dataset paths in the **read_data.py** file to point to the locally stored datasets. Then, run the **model_train.py** file to initiate the entire training process.

First, the **HGAO** (Hybrid Giant Armadillo Optimization) algorithm is invoked for hyperparameter tuning. Through multiple iterations, this algorithm intelligently searches for the global optimal solution, continuously optimizing the hyperparameters within the search space. It ultimately outputs the optimal hyperparameter combination and its corresponding fitness value.

Once HGAO completes the optimization and generates the best hyperparameters, the system automatically passes these parameters to the **DenseNet-121** model. At this point, the model begins training, and the process continues until the model reaches convergence, ensuring maximum performance.

Upon completion of the training, the system will conduct model testing, outputting key metrics such as test accuracy, test loss, and additional metrics including precision, recall, and F1-score. These metrics validate the performance of the DenseNet-121 model on the target task. This comprehensive workflow ensures efficient model training and evaluation, achieving optimal classification performance.



## Environment configuration

Python 3.8 environment using Anaconda with the following configuration:

| library        | version  |
| -------------- | -------- |
| cuda           | 11.2     |
| cudnn          | 8.2      |
| tensorflow-gpu | 2.10     |
| keras          | 2.10     |
| scikit-learn   | 1.3.2    |
| opencv-python  | 4.9.0.80 |

The computational experiments were conducted on a system equipped with an Intel® Xeon® Gold 6142 CPU, operating at 2.60 GHz, and supplemented with 64.0 GB of internal storage. The system also featured an NVIDIA RTX A5000 GPU, which was utilized for accelerated processing tasks.
