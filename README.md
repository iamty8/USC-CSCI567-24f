# A CSCI-567 Project on Novel Class Discovery


Novel Class Discovery is a machine learning task where the goal is to identify new, unseen classes from unlabeled data, using knowledge from a set of known, labeled classes. The challenge is to learn effective representations that can generalize to the novel classes, leveraging the similarities between known and unknown classes, without direct supervision for the new categories.

While image classification works well and has an extremely high performance on lots of current popular datasets, novel class discovery remains challenging for these same datasets and the classification performance significantly when it comes to unknown classes.


## Running

### Dependencies

```
conda create -n simgcd python==3.9
activate simgcd
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```

### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

We found picking the model according to 'Old' class performance could lead to possible over-fitting, and since 'New' class labels on the held-out validation set should be assumed unavailable, we suggest not to perform model selection, and simply use the last-epoch model.

## Datasets

### CIFAR-100
Nowadays, simply classifying [CIFAR-100](https://pytorch.org/vision/stable/datasets.html) into predefined a hundred types could already achieve a high accuracy above 95% using swin transformers and other large vision models. However, it is still not easy to classify CIFAR100 under the novel class discovery setting (most works only achieve an accuracy of around 70-80%).

Specifically, in CIFAR-100 dataset, we follow previous work settings to have 80 labelled classes and 20 unlabeled classes, and then we test the model performance both on old categories and new categories.

### ODIR-5K

[ODIR-5K](https://odir2019.grand-challenge.org/dataset/) is a structured ophthalmic dataset collected by Shanggong Medical Technology Co., Ltd., containing real-life patient information from various hospitals and medical centers in China. It includes data from 5,000 patients, with details such as age, color fundus photographs of both left and right eyes, and diagnostic keywords provided by doctors.

The dataset contains color fundus photographs captured by different cameras available in the market, such as Canon, Zeiss, and Kowa, resulting in varied image resolutions.

The annotations are provided by trained human readers under quality control management. Patients are categorized into eight labels based on their eye images and age:
+ Normal (N)
+	Diabetes (D)
+	Glaucoma (G)
+	Cataract (C)
+	AMD (A)
+	Hypertension (H)
+	Myopia (M)
+	Other Diseases/Abnormalities (O)

## Related Works
We’ll be following [Parametric Classification for Generalized Category Discovery: A Baseline Study (ICCV 2023)](https://github.com/CVMI-Lab/SimGCD) as our baseline, reproducing the algorithm and results from the paper, and attempting to improve the related performance on CIFAR-100 below:

<table><thead><tr><th>Source</th><th colspan="3">Paper (3 runs) </th><th colspan="3">Current Github (5 runs) </th></tr></thead><tbody><tr><td>Dataset</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>CIFAR10</td><td>97.1±0.0</td><td>95.1±0.1</td><td>98.1±0.1</td><td>97.0±0.1</td><td>93.9±0.1</td><td>98.5±0.1</td></tr><tr><td>CIFAR100</td><td>80.1±0.9</td><td>81.2±0.4</td><td>77.8±2.0</td><td>79.8±0.6</td><td>81.1±0.5</td><td>77.4±2.5</td></tr><tr><td>ImageNet-100</td><td>83.0±1.2</td><td>93.1±0.2</td><td>77.9±1.9</td><td>83.6±1.4</td><td>92.4±0.1</td><td>79.1±2.2</td></tr><tr><td>ImageNet-1K</td><td>57.1±0.1</td><td>77.3±0.1</td><td>46.9±0.2</td><td>57.0±0.4</td><td>77.1±0.1</td><td>46.9±0.5</td></tr><tr><td>CUB</td><td>60.3±0.1</td><td>65.6±0.9</td><td>57.7±0.4</td><td>61.5±0.5</td><td>65.7±0.5</td><td>59.4±0.8</td></tr><tr><td>Stanford Cars</td><td>53.8±2.2</td><td>71.9±1.7</td><td>45.0±2.4</td><td>53.4±1.6</td><td>71.5±1.6</td><td>44.6±1.7</td></tr><tr><td>FGVC-Aircraft</td><td>54.2±1.9</td><td>59.1±1.2</td><td>51.8±2.3</td><td>54.3±0.7</td><td>59.4±0.4</td><td>51.7±1.2</td></tr><tr><td>Herbarium 19</td><td>44.0±0.4</td><td>58.0±0.4</td><td>36.4±0.8</td><td>44.2±0.2</td><td>57.6±0.6</td><td>37.0±0.4</td></tr></tbody></table>

## Task Expansion

After we revise the algorithm and verify our correctness about improving the performance on CIFAR100 dataset, if time allows, we also want to test our algorithm on other datasets. Now we are specifically interested in the following medical dataset:

## Implementation Detail

Something

## Our Results

The results go here.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
