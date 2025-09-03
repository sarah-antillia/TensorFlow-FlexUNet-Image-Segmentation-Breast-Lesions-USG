<h2>TensorFlow-FlexUNet-Image-Segmentation-Breast-Lesions-USG (2025/09/03)</h2>

This is the first experiment of Image Segmentation for Breast-Lesions-USG (Benign and Malignant)
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1Lz_95ooDFiIkXHSO7s3CdeSnt8y7LgGI/view?usp=sharing">
Augmented-Breast-Lesions-USG-ImageMask-Dataset.zip</a> with colorized masks (benign:green, malignant:red), 
which was derived by us from 
<br><br>
<a href="https://www.cancerimagingarchive.net/collection/breast-lesions-usg/">
<b>
Breast-Lesions-USG | A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis
</b>
</a>
<br>

<br>
<b>Acutual Image Segmentation for 512x512 Breast-Lesions-USG images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br>
<b>rgb_map =  (benign:green, malignant:red)</b><br>
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10013.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10058.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10058.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10058.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10171.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10171.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10171.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been taken from <b>The Cancer Imaging Archive</b>
<br>
<br>
<a href="https://www.cancerimagingarchive.net/collection/breast-lesions-usg/">
<b>
Breast-Lesions-USG | A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis
</b>
</a>
<br><br>
<b>Summary</b><br>

This dataset consists of 256 breast ultrasound scans collected from 256 patients and 266 benign and malignant segmented lesions.  
It includes patient-level labels, image-level annotations, and tumor-level labels with all cases confirmed by follow-up care 
or biopsy result. Each scan was manually annotated and labeled by a radiologist experienced in breast ultrasound examination. 
In particular, each tumor was identified in the image via a freehand annotation and labeled according to BIRADS features. 
The tumor histopathological classification is stated for patients who underwent a biopsy. 
Patient-level labels include clinical data such as age, breast tissue composition, signs and symptoms. 
<br>
Image-level freehand annotations identify the tumor and other abnormal areas in the image. <br>
The tumor and image are labeled with BIRADS category, 7 BIRADS descriptors, and interpretation of critical 
findings as presence of breast diseases. <br>
Additional labels include the method of verification, tumor classification and histopathological diagnosis.
<br>
Since the role of machine learning and theoretical computing towards the development of augmented inference 
in the field of cancer detection is indisputable, the quality of the data used to develop any explainable 
augmented inference methods is extremely important. This dataset can be used as an external testing set 
for assessing a model’s performance and for developing explainable AI or supervised machine learning
 models for the detection, segmentation and classification of breast abnormalities in ultrasound images.
<br><br>
A detailed description of this dataset can be found here and should be cited along with the citation of the data:
<br>
Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A.,
 & Zolek, N.<br>
  A Curated benchmark dataset for ultrasound based breast lesion analysis. Sci Data 11, 148 (2024).<br>
   https://doi.org/10.1038/s41597-024-02984-z.
<br>
<br>
<b>Citations & Data Usage Policy</b><br>
 Data Citation Required: Users must abide by the TCIA Data Usage Policy and Restrictions.<br>
  Attribution must include the following citation, including the Digital Object Identifier:
<br><br>
<b>Data Citation</b><br>
Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł.,<br>
 Nowicki, A., & Zolek, N. (2024). <br>
 A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis (Breast-Lesions-USG) (Version 1) <br>
 [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9WKK-Q141
<br>
<br>
<h3>
<a id="2">
2 Augmented-Breast-Lesions-USG ImageMask Dataset
</a>
</h3>
 If you would like to train this Breast-Lesions-USG Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1Lz_95ooDFiIkXHSO7s3CdeSnt8y7LgGI/view?usp=sharing">
Augmented-Breast-Lesions-USG-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Breast-Lesions-USG
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Breast-Lesions-USG Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/Breast-Lesions-USG_Statistics.png" width="512" height="auto"><br>
<br>
<!--
On the derivation of the 512x512 pixels augmented dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>,
and a tumor classification file. 
<li><a href="./generator/BrEaST-Lesions-USG-clinical-data-Dec-15-2023-Classification.csv">
BrEaST-Lesions-USG-clinical-data-Dec-15-2023-Classification.csv</a></li>
,which was derived from the original
<a href="https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx">
BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
</a>
<br>
-->
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Breast-Lesions-USG TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Breast-Lesions-USG and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Breast-Lesions-USG 1+2 classes.<br>
<pre>
[mask]
mask_file_format = ".png"

; RGB colors    benign:green, malignanat:red    
rgb_map = {(0,0,0):0,(0,255,0):1,(255,0,0):2,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 50 by EearlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/train_console_output_at_epoch50.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Breast-Lesions-USG</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Breast-Lesions-USG.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/evaluate_console_output_at_epoch50.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Breast-Lesions-USG/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0368
dice_coef_multiclass,0.9835
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Breast-Lesions-USG</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Breast-Lesions-USG.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<b>rgb_map =  (benign:green, malignant:red)</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10080.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10080.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10080.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10142.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10142.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10142.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/10198.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/10198.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/10198.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/barrdistorted_1001_0.3_0.3_10155.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/barrdistorted_1001_0.3_0.3_10155.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/barrdistorted_1001_0.3_0.3_10155.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/barrdistorted_1001_0.3_0.3_10163.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/barrdistorted_1001_0.3_0.3_10163.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/barrdistorted_1001_0.3_0.3_10163.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/images/barrdistorted_1001_0.3_0.3_10207.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test/masks/barrdistorted_1001_0.3_0.3_10207.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Lesions-USG/mini_test_output/barrdistorted_1001_0.3_0.3_10207.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Breast-Lesions-USG | A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis
</b><br>
<a href="https://www.cancerimagingarchive.net/collection/breast-lesions-usg/">
https://www.cancerimagingarchive.net/collection/breast-lesions-usg/</a>

<br><br>
<b>2. Curated benchmark dataset for ultrasound based breast lesion analysis</b>
Anna Pawłowska, Anna Ćwierz-Pieńkowska, Agnieszka Domalik, Dominika Jaguś, <br>
Piotr Kasprzak, Rafał Matkowski, Łukasz Fura, Andrzej Nowicki & Norbert Żołek <br>
<a href="https://www.nature.com/articles/s41597-024-02984-z">
https://www.nature.com/articles/s41597-024-02984-z
</a>
<br>
<br>

<b>3. TensorFlow-FlexUNet-Image-Segmentation-BUS-BRA</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BUS-BRA">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BUS-BRA
</a>
<br>
<br>

<b>4. TensorFlow-FlexUNet-Image-Segmentation-BUS-UCLM</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BUS-UCLM">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BUS-UCLM
</a>
<br>
<br>

<b>5. TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Ultrasound-Images
</a>
<br>
<br>

