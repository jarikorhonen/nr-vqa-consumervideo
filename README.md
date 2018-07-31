# nr-vqa-consumervideo
# Hierarchical Video Quality Model (HiViQuM)

This work implements No-Reference Video Quality Model (NR-VQM) intended specially for assessing consumer video quality, 
impaired typically by capture artifacts, such as sensor noise, motion blur and camera shakiness. Most of the NR-VQMs known
in the prior art have been designed and optimized for compression and transmission artifacts, and therefore they are not
optimal for consumer video. Another problem with many of the known NR-VQMs is their high complexity, which limits their use
in practical consumer applications.

More information about the proposed model is available in the publication (under review).

The workflow for using the software consists of two parts: 1) Feature extraction, and 2) Learning-based regression. The
implementation of feature extraction is in the source file _compute_nrvqa_features.m_ (Matlab). An example showing how to
extract the features for videos from KoNViD-1k database and write them in a comma separated value (CSV) file is in
_compute_features_example.m_. For using the example, you need to download the KoNViD-1k video sequences and the MOS scores
from there, and decode the video sequences in raw YUV format. An example in _train_and_validate_KoNViD.py_ file shows how
the regression model for predicting the MOS values from the features can be trained and validated.
