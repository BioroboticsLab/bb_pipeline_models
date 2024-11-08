import tensorflow as tf

# Access Keras layers and models from tf.keras
Input = tf.keras.layers.Input
SpatialDropout2D = tf.keras.layers.SpatialDropout2D
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
Activation = tf.keras.layers.Activation
Add = tf.keras.layers.Add
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dense = tf.keras.layers.Dense
Model = tf.keras.models.Model

import keras
import numpy as np
import pickle
import json


## localizer model

def get_conv_model(initial_channels=32):
    inputs = Input(shape=(None, None, 1))

    x_0 = Conv2D(initial_channels, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x_0 = SpatialDropout2D(.1)(x_0)

    x_1 = Conv2D(initial_channels * 2 ** 1, (3, 3), strides=(2, 2), activation='relu')(x_0)
    x_1 = BatchNormalization()(x_1)
    x_1 = SpatialDropout2D(.1)(x_1)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(2, 2), activation='relu')(x_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)

    x_3 = Conv2D(initial_channels * 2 ** 3, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_3 = BatchNormalization()(x_3)

    xu_3 = Conv2D(initial_channels, (1, 1), strides=(1, 1), activation='relu')(x_3)
    xu_3 = Conv2D(4, (1, 1), activation='sigmoid')(xu_3)

    model = Model(inputs=inputs, outputs=xu_3)

    return model


## decoder: custom resnet model


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2a', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', 
               name=conv_name_base + '2b', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2c', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', 
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', 
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', 
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', 
                        kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def get_custom_resnet(input_shape=(32, 32, 1), classes=1):
    X_input = Input(input_shape, name="input_2")

    # Stage 1 (Res2)
    X = Conv2D(16, (1, 1), strides=(1, 1), padding='valid', name='res2a_branch2a', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn2a_branch2a')(X)
    X = Activation('relu', name='activation_150')(X)
    
    X_shortcut = Conv2D(64, (1, 1), strides=(1, 1), name='res2a_branch1', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_input)
    X_shortcut = BatchNormalization(axis=3, name='bn2a_branch1')(X_shortcut)
    
    X = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='res2a_branch2b', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn2a_branch2b')(X)
    X = Activation('relu', name='activation_151')(X)
    
    X = Conv2D(64, (1, 1), strides=(1, 1), name='res2a_branch2c', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn2a_branch2c')(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu', name='activation_152')(X)
    
    X = identity_block(X, 3, [16, 16, 64], stage=2, block='b')

    # Stage 2 (Res3)
    X = convolutional_block(X, f=3, filters=[32, 32, 128], stage=3, block='a', s=2)
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='b')

    # Stage 3 (Res4)
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=4, block='a', s=2)
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='b')

    # Stage 4 (Res5)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=5, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=5, block='b')

    # Stage 5 (Res6)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=6, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=6, block='b')

    # AVGPOOL
    X = GlobalAveragePooling2D(name='avg_pool')(X)

    # Output layers
    bit_outputs = [Dense(1, activation='sigmoid', name=f'bit_{i}')(X) for i in range(12)]
    rotation_outputs = [Dense(2, activation='linear', name=f'{axis}_rotation')(X) for axis in ['x', 'y', 'z']]
    center_output = Dense(2, activation='linear', name='center')(X)

    # Create model
    model = Model(inputs=X_input, outputs=bit_outputs + rotation_outputs + [center_output], name='custom_resnet50')

    return model


# function for comparing model structures, ignoring the names of the layers but comparing the type, shape, and number of parameters
# also compare the predictions of the models on randomly generated test data

def compare_models(model1, model2, tol=1e-5, num_samples=10):
    # Use a fallback input shape if input shape is not fully defined
    if model1.input_shape[1:] is None or any(dim is None for dim in model1.input_shape[1:]):
        input_shape = (1000, 1000, 1)  # Default shape
        print(f"Using default input shape: {input_shape}")
    else:
        input_shape = model1.input_shape[1:]  # Ignore the batch size

    # Generate synthetic test data based on the model's expected or fallback input shape
    test_data = np.random.rand(num_samples, *input_shape).astype(np.float32)

    # Compare architectures ignoring layer names
    for layer1, layer2 in zip(model1.layers, model2.layers):
        if type(layer1) != type(layer2):
            print(f"Different layer types: {type(layer1)} vs {type(layer2)}")
            return False
        if layer1.output_shape != layer2.output_shape:
            print(f"Different output shapes: {layer1.output_shape} vs {layer2.output_shape}")
            return False
        if layer1.count_params() != layer2.count_params():
            print(f"Different number of parameters: {layer1.count_params()} vs {layer2.count_params()}")
            return False

    # Compare weights
    for layer1, layer2 in zip(model1.layers, model2.layers):
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        for w1, w2 in zip(weights1, weights2):
            if not np.array_equal(w1, w2):
                print("The weights are different in layer:", layer1.name)
                return False

    # Compare predictions on generated test data
    predictions1 = model1.predict(test_data)
    predictions2 = model2.predict(test_data)
    
    # Check each prediction individually for closeness
    for i, (p1, p2) in enumerate(zip(predictions1, predictions2)):
        if not np.allclose(p1, p2, atol=tol):
            print(f"Prediction mismatch at index {i} with values:\nModel1: {p1}\nModel2: {p2}")
            return False

    print("The models are identical in architecture, weights, and predictions.")
    return True


def SaveModelData(model, test_data, file_prefix):
    """
    Save the model architecture, weights, and predictions in a single .pkl file.
    
    Parameters:
    - model: The Keras model to save data from.
    - test_data: numpy array, the test data to use for generating predictions.
    - file_prefix: string, the prefix for the output file.
    """
    # Save model architecture
    model_json = model.to_json()
    
    # Save model weights as numpy arrays
    weights = model.get_weights()
    
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Compile all data into a dictionary
    model_data = {
        'architecture': model_json,
        'weights': weights,
        'test_data': test_data,
        'predictions': predictions
    }
    
    # Save all model data into a single pickle file
    with open(f'{file_prefix}_model_data.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model data saved to '{file_prefix}_model_data.pkl'")


def CompareModelData(file1, file2, tol=1e-5):
    """
    Compare the saved model data from two files, ignoring layer names and normalizing 'None' values.
    
    Parameters:
    - file1: string, the path to the first model data file.
    - file2: string, the path to the second model data file.
    - tol: float, the tolerance level for comparing numerical values.
    """
    # Load data from both files
    with open(file1, 'rb') as f:
        data1 = pickle.load(f)
    with open(file2, 'rb') as f:
        data2 = pickle.load(f)
    
    # Parse architectures
    layers1 = json.loads(data1['architecture'])['config']['layers']
    layers2 = json.loads(data2['architecture'])['config']['layers']
    
    # Check number of layers
    if len(layers1) != len(layers2):
        print("Different number of layers in architectures.")
        return False
    
    # Compare architecture by layer type and output shape, normalizing None values
    for i, (layer1, layer2) in enumerate(zip(layers1, layers2)):
        if layer1['class_name'] != layer2['class_name']:
            print(f"Different layer types at index {i}: {layer1['class_name']} vs {layer2['class_name']}")
            return False

        # Normalize None values by treating shapes with None as equivalent
        shape1 = layer1['config'].get('batch_input_shape')
        shape2 = layer2['config'].get('batch_input_shape')
        
        if shape1 and shape2:
            shape1 = tuple(dim if dim is not None else -1 for dim in shape1)
            shape2 = tuple(dim if dim is not None else -1 for dim in shape2)
            if shape1 != shape2:
                print(f"Different output shapes at layer {i}: {shape1} vs {shape2}")
                return False

    print("Model architectures are identical in layer types and output shapes.")
    
    # Compare weights by ignoring layer names
    weights1 = data1['weights']
    weights2 = data2['weights']
    
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if not np.allclose(w1, w2, atol=tol):
            print(f"Weights differ at index {i}.")
            return False
    print("Model weights are identical.")
    
    # Compare predictions
    predictions1 = data1['predictions']
    predictions2 = data2['predictions']
    if isinstance(predictions1, list):
        for i, (p1, p2) in enumerate(zip(predictions1, predictions2)):
            if not np.allclose(p1, p2, atol=tol):
                print(f"Predictions differ at output {i}.")
                return False
    else:
        if not np.allclose(predictions1, predictions2, atol=tol):
            print("Predictions differ.")
            return False
    print("Model predictions are identical.")
    
    print("All comparisons passed.")
    return True