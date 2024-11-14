# bb_pipeline_models

This repository contains models used for running detection, saliency (localizer), and tracking in the `bb_pipeline`. Models are provided for compatibility with both Keras 2 and Keras 3.

## Model Files and Compatibility

The following is a summary of the models, their purpose, and their compatibility with different versions of Keras.

### Detection Step

#### Localizer Model (Saliency)

- **Keras 3 Version**: `localizer_2019_keras3.h5` and `localizer_2019_attributes.json`  
  - Additional attributes (`localizer_2019_attributes.json`) are required for using the model in Keras 3.
- **Keras 2 Version**: `.model` files (e.g., `localizer_2019.model`)

#### Decoder Model

- **Keras 3 Version**: `decoder_2019_keras3.h5`
- **Keras 2 Version**: `decoder_2019.model`

### Tracking Step

- **Model Files**: `detection_model_4.json`, `tracklet_model_8.json`
  - These tracking model files in JSON format are compatible with both Keras 2 and Keras 3.

## Folder Structure and Model Generation

### `models` Folder

This folder contains the models used in the pipeline, organized by task:

- **`decoder`**: Contains models for the detection step.
- **`saliency`**: Contains models for the saliency (localizer) step.
- **`tracking`**: Contains tracking models compatible with both Keras 2 and Keras 3.

### `model_generation` Folder

This folder provides notebooks and code for model conversion and generation:

- **`keras2_model_conversion.ipynb`**: Notebook to convert existing Keras 2 models to a format compatible with Keras 3 by saving the model weights.
- **`keras3_model_generation.ipynb`**: Notebook with functions to create model structures compatible with Keras 3 by importing model weights.
- **`pipelinemodels.py`**: Contains the model structure functions, which generate models for the decoder and localizer steps.

