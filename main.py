
import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt



# LOAD

def load_data_tflite_model_maker(path, split_trainrest_ratio=0.9, split_testval_ratio=0.5):
    data = DataLoader.from_folder(path, shuffle=True)
    train_data, rest_data = data.split(split_trainrest_ratio)
    validation_data, test_data = rest_data.split(split_testval_ratio)

    return train_data, test_data, validation_data


# Normalization
''' Normalization of images '''
#normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
#normalized_ds = train_data.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
#print(np.min(first_image), np.max(first_image))

'''  USE DATA AUGMENTATION  '''
#data_augmentation = tf.keras.Sequential([
#    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
#])


# Build
def build_model(
    train_d, 
    validation_data, 
    model_spec='efficientnet_lite0',
    batch_size=32, 
    epochs=5, 
    train_whole_model=False, 
    dropout_rate=0.2,
    learning_rate=0.005, 
    momentum=0.9, 
    shuffle=False, 
    use_augmentation=False,
    use_hub_library=True, 
    warmup_steps=None, 
    model_dir="./model_dir/",
    do_train=True ):
    
    model = image_classifier.create(
       train_d,
        model_spec=model_spec,                  validation_data=validation_data,
        batch_size=batch_size,                  epochs=epochs,
        train_whole_model=train_whole_model,    dropout_rate=dropout_rate,              
        learning_rate=learning_rate,            momentum=momentum,                      
        shuffle=shuffle,                        use_augmentation=use_augmentation,      
        use_hub_library=use_hub_library,        warmup_steps=warmup_steps,              
        model_dir=model_dir,                    do_train=do_train
    )

    return model

def plot_classified_images(model, test_data):

    # Helper function that returns "red or black" depending on if its two input parameters mathces or not
    def get_label_color(val1, val2):
        if val1 == val2:
            return 'black'
        else:
            return 'red'

    # Then plot 100 test images and their predicted labels. If a prediciton result is 
    # different from the label provided label in "test" dataset, we will highligh it in red color.
    plt.figure(figsize=(20.0, 20.0)) # Needs to force Float numbers.
   
    #random_images = test_data.split(0.99) # they are shuffled prior to the creation of test_data, therefore we know that they are not the same as last time
    #super_random_images = super(ImageClassifierDataLoader.__init__(random_images, random_images.__len__(), 2 ))
    predicts = model.predict_top_k(test_data, k=2, batch_size=32)   #   def predict_top_k(self, data, k=1, batch_size=32)
    #xfx = super_random_images.dataset.take(100)
  
    for i, (image, label) in enumerate(test_data._dataset.take(100)):           #enumerate( test_data.dataset.take(100)):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)

        predict_label = predicts[i][0][0]
        color = get_label_color(predict_label, test_data.index_to_label[label.numpy()])         #test_data.index_to_label[label.numpy()])
        ax.xaxis.label.set_color(color)
        plt.xlabel('Predicted: %s' % predict_label)
    plt.show()

# Evaluate 
def evaluate_model(test_data, model):
    loss, accuracy = model.evaluate(test_data)
    return loss, accuracy


# Export model
def export_model(model, export_dir, overwrite, export_format, label_filename, save_model_filename):
    model.export(
        export_dir=export_dir,        #, tflite_filname='sheep_classifcation_model.tflite'
        export_format=export_format, 
        label_filename=label_filename,
        overwrite=overwrite,
        save_model_filename=save_model_filename
    )

# Alternative Evaluation method #TODO: Not working
def alternative_evaluation_tflite_mode(model, test_data):
    model.evaluate_tflite('model.tflite', test_data)


# Load
#module_spec = hub.load_module_spec("./model_files/")
#height, width = hub.get_expected_image_size(module_spec)
#images = ...  # A batch of images with shape [batch_size, height, width, 3].
#module = hub.Module(module_spec)
#features = module(images)   # A batch with shape [batch_size, num_features].
# Build

# Evaluate

# Export model

# Draw learning curves
def draw_learning_curves(model):
    plt.plot(model.hitory['loss'], color='b')
    plt.plot(model.history['val_loss'], color='r')
    plt.show()

    plt.plot(model.hitory['binary_accuracy'], color='b')
    plt.plot(model.hitory['val_binary_accuracy'], color='r')
    plt.show()

    plt.plot(model.hitory['mean_square_error'], color='b')
    plt.plot(model.hitory['val_mean_squared_error'], color='r')
    plt.show()

if __name__ == '__main__':

    path = '../saue bilder/Combined/Visual_originale/'
    path_2 = '/mnt/GigaServer1/Haakosbo/bilder/Visual_originale/'
    path_3 = '../Visual_originale/'
    path_4 = '../saue bilder/Combined/Visual/'

    path = path_4


    split_trainrest_ratio = 0.8
    split_testval_ratio = 0.5
    
    model_spec = 'efficientnet_lite4' # 'efficientnet_lite0-4 'mobilenet_v2' 'resnet_50'
    epochs = 300
    dropout_rate = 0.5
    learning_rate = None
    shuffle = True
    batch_size = None
    use_augmentation = True
    train_whole_model = True

    # Load Data
    train_data, test_data, validation_data = load_data_tflite_model_maker(path, split_trainrest_ratio, split_testval_ratio)

    #test = train_data._dataset.take(2)
    #test_2 = test_data._dataset.take(2)

    # Build model 
    #modelx_ = image_classifier.create(train_data)
    model = build_model(
        train_d=train_data, 
        validation_data=validation_data, 
        model_spec=model_spec, 
        epochs=epochs,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        train_whole_model=train_whole_model,
        )
    model.train( train_data, validation_data, hparams=None)
    model.summary()

    # Evaluate model
    loss, accuracy = evaluate_model(test_data=test_data, model=model)

    #plot classified pictures "Manual checking"
    xx = model.train(train_data)
    xx.history
    print(xx.history)
    #model.predict_top_k(test_data)
    #test_data
    #plot_classified_images(model=model, test_data=test_data)

    # Export model
    export_model(
        model, 
        export_dir='model_files/', 
        label_filename='classification_labels', 
        overwrite=True,
        export_format=ExportFormat.SAVED_MODEL,
        save_model_filename=str('tflite_model_saved')

        )
    #draw_learning_curves(model)

    print("done done")
    quit()













    ''' GRAVEYARD 


#from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
#from tensorflow_examples.lite.model_maker.core.task.image_classifier import ImageClassifier
#import tensorflow_hub as hub
#from tensorflow_examples.lite.model_maker.core import export_format
#from tensorflow_hub.tools.make_image_classifier.make_image_classifier_lib import make_image_classifier
#from tflite_model_maker.image_classifier import ModelSpec
#from tflite_model_maker.config import QuantizationConfig
#import os
#import numpy as np
#from sklearn.utils import validation



 def build_model(train_data, validation_data, model_spec='efficientnet_lite0', epochs=5, do_fine_tuning=False, 
        batch_size=32, learning_rate=0.005, momentum=0.9, dropout_rate=0.2, l1_regularizer=0.0, l2_regularizer=0.0001,
        label_smoothing=0.1, validation_split=0.2, do_data_augmentation=False, rotation_range=40, horizontal_flip=True, 
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2):
    
        'efficientnet_lite0' 'efficientnet_lite1' 'efficientnet_lite2'
        'efficientnet_lite3' 'efficientnet_lite4' 
        'mobilenet_v2' 'resnet_50'
        train_data,
             model_spec='efficientnet_lite0',
             validation_data=None,
             batch_size=None,
             epochs=None,
             train_whole_model=None,
             dropout_rate=None,
             learning_rate=None,
             momentum=None,
             shuffle=False,
             use_augmentation=False,
             use_hub_library=True,
             warmup_steps=None,
             model_dir=None,
             do_train=True
   
    model = image_classifier.create(
        train_data=train_data,                  model_spec=model_spec, 
        validation_data=validation_data,        epochs=epochs, 
        do_fine_tuning=do_fine_tuning,          batch_size=batch_size, 
        learning_rate=learning_rate,            momentum=momentum, 
        dropout_rate=dropout_rate,              l1_regularizer=l1_regularizer, 
        l2_regularizer=l2_regularizer,          label_smoothing=label_smoothing, 
        validation_split=validation_split,      do_data_augmentation=do_data_augmentation, 
        rotation_range=rotation_range,          horizontal_flip=horizontal_flip, 
        width_shift_range=width_shift_range,    height_shift_range=height_shift_range, 
        shear_range=shear_range,                zoom_range=zoom_range       )

    return model

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 '''