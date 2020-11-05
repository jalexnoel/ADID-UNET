# ADID-UNET-A-Segmentation-model-for-COVID-19-Lung-Infection-from-CT-Scans
A new segment depth network for COVID-19 lung CT scans, Attention Gate-Dense Network- Improved Dilation Convolution-UNET (ADID-UNET)

Now we just only upload the trained model and the usage document, we will upload all the codes and test data here as well later. Now we would like to explain how to run the codes to obtain the predicted results and segmentation indexes firstly:

If you get the codes, please prepare the datasets, then: 
1. Use aug.py to augment the data, and then use the data.py to convert the data to .NPY format.
2. Use unetdeeplidate.py to train the ADID-UNET model.
3. Use predict.py and plotone.py to obtain the prediction results and segmentation indicators, such as accuracy, precision, Dice coefficient, sensitivity, specificity and F1 score. 
4. Use matlab script to obtain the other three segmentation indexes, Structural metric (Sm), Enhance alignment metric (Eα), Mean Absolute Error (MAE).

