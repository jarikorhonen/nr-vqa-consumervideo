# Two Level Video Quality Model (TLVQM)

![HiViQuM diagram](https://github.com/jarikorhonen/nr-vqa-consumervideo/blob/master/hiviqum.png)

This work implements No-Reference Video Quality Model (NR-VQM) intended specially for assessing consumer video quality, 
impaired typically by capture artifacts, such as sensor noise, motion blur and camera shakiness. Most of the NR-VQMs known
in the prior art have been designed and optimized for compression and transmission artifacts, and therefore they are not
optimal for consumer video. Another problem with many of the known NR-VQMs is their high complexity, which limits their use
in practical consumer applications. The aim of the proposed model is to predict subjective video quality as accurately as
possible, even in the presence of capture artifacts, with a reasonable computational complexity.

More information about the proposed model will available in the related paper (Early Access version available [here](https://ieeexplore.ieee.org/document/8742797)).

The workflow for using the software consists of two parts: 1) Feature extraction, and 2) Learning-based regression. The
implementation of feature extraction is in the source file _compute_nrvqa_features.m_ (Matlab). An example showing how to
extract the features for videos from LIVE-Qualcomm database and write them in a comma separated value (CSV) file is in
*compute_features_example.m*. For using the example, you need to download the LIVE-Qualcomm video sequences (raw YUV) 
and the related MOS scores from [there](http://live.ece.utexas.edu/research/incaptureDatabase/index.html). An example in *train_and_validate_example.py* file shows how the regression model for predicting the MOS values from the features can be 
trained and validated (this part is in Python, using scikit-learn toolbox for regression).
