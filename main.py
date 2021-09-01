
import os as os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.eager.context import PhysicalDevice

import tflite_model_maker as tflite
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from datetime import datetime



tf.config.threading.set_inter_op_parallelism_threads(24)
tf.config.threading.set_intra_op_parallelism_threads(24)


'''
    Dictonary to hold the paths to different folders of preprocessing been done.
'''
def return_image_path_dict():
    return {
        #'image_path_IR'                         : '../saue bilder/Combined/IR_originale/',
        #'img_path_IR_duplicate_STRICT'          : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_STRICT/',
        #'img_path_IR_duplicate_BASIC'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_BASIC/',
        #'img_path_IR_duplicate_LOOSE'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_LOOSE/',
        
        #'img_path_IR_BLURRY_duplicate_STRICT'   : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_STRICT/',
        #'img_path_IR_BLURRY_duplicate_BASIC'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_BASIC/',
        #'img_path_IR_BLURRY_duplicate_LOOSE'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_LOOSE/',
        'img_path_VISUAL'                        : '../saue bilder/Combined/Visual/',
        'img_path_VISUAL_duplicate_STRICT'      : '../saue bilder/Combined/Visual_removed_duplicate_STRICT/',
        'img_path_VISUAL_duplicate_BASIC'       : '../saue bilder/Combined/Visual_removed_duplicate_BASIC/',
        'img_path_VISUAL_duplicate_LOOSE'       : '../saue bilder/Combined/Visual_removed_duplicate_LOOSE/'
        }

'''
    Loads data according tflite's imageclassifiers specification
'''
def load_data_tflite_model_maker(path, split_trainrest_ratio=0.9, split_testval_ratio=0.5):
    data = image_classifier.DataLoader.from_folder(path, shuffle=True)
    train_data, rest_data = data.split(split_trainrest_ratio)
    validation_data, test_data = rest_data.split(split_testval_ratio)

    return train_data, test_data, validation_data


'''
    Build the model with the necessary inputs expected in the tflite-model-maker env
    model_dir/ represnet the checkpoint data are tempory saved between epochs of the network.
    returns a model object.
'''
def build_model(
    train_data, 
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

    # calls the image_classifier Create with each specified input
    model = image_classifier.create(
        train_data,
        model_spec=model_spec,                  
        validation_data=validation_data,
        batch_size=batch_size,                  
        epochs=epochs,
        train_whole_model=train_whole_model,    
        dropout_rate=dropout_rate,              
        learning_rate=learning_rate,            
        momentum=momentum,                      
        shuffle=shuffle,                        
        use_augmentation=use_augmentation,      
        use_hub_library=use_hub_library,        
        warmup_steps=warmup_steps,              
        model_dir=model_dir,                   
        do_train=do_train
    )
    return model

'''
    Evaluate the model based on test_data passed in. 
    Important that this data is validation data and have
    not been "seen" before. 
    returns two arrays of loss and accuracy
'''
def evaluate_model(test_data, model):
    loss, accuracy = model.evaluate(test_data)
    return loss, accuracy

'''Exports the model, to specified folder'''
def export_model(model, export_dir, overwrite, export_format, label_filename, save_model_filename):
    model.export(
        export_dir=export_dir,        #, tflite_filname='sheep_classifcation_model.tflite'
        export_format=export_format, 
        label_filename=label_filename,
        overwrite=overwrite,
        save_model_filename=save_model_filename
    )

'''Based on a guide within tensorflow. 
    The def plots the images based on if they where correctly labeled
    or not. Red color incicates a wrong prediciton.'''
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

'''
    Draws a matplotlib curves of the training based on the objects history.
    the plot is saved with a unique name in a specific folder.
'''
def draw_learning_curves(history, model_title):
    #plot_title = str(path.split("/")[-2])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    figure = plt.figure(figsize=(8, 8))
    plt.ion()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    #plt.close(figure)
    plt.savefig("plots/"+ model_title + "_" + str(len(acc)) + ".png")
    #plt.show()

'''
    Findds the local time
'''
def get_local_time():
    d = datetime.now()
    return str(d.day) + "." + str(d.month) + " " + str(d.hour) + "." + str(d.minute)

'''
    Creats a new directory where newly created models can be
    saved, avoid model data being saved in same folder upon multiple
    runs. Returns the name of the folder.
'''
def create_new_dir(model_name, epochs):
    local_time = get_local_time()
    new_dir = str(model_name) + "_" + str(epochs) + "_epochs_" + str(local_time) 
    parent_dir = "./model_files/"
    path = os.path.join(parent_dir, new_dir)
    os.mkdir(path)
    return new_dir #path


if __name__ == '__main__':
    
    # Loop over dictonary to target each path
    for i, (k, v) in enumerate(return_image_path_dict().items()):
        # rename v to path, to easier readability.
        path = v 
        
        split_trainrest_ratio = 0.7
        split_testval_ratio = 0.5
        model_spec = 'efficientnet_lite4' # 'efficientnet_lite0-4 'mobilenet_v2' 'resnet_50'
        epochs = 500
        dropout_rate = 0.5
        learning_rate = 0.002
        shuffle = True
        batch_size = None
        use_augmentation = True
        train_whole_model = False

        # Load Data
        train_data, test_data, validation_data = load_data_tflite_model_maker(path, split_trainrest_ratio, split_testval_ratio)

        # Build model 
        model = build_model(
            train_data=train_data, 
            validation_data=validation_data, 
            model_spec=model_spec, 
            epochs=epochs,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            train_whole_model=train_whole_model,
            )

        # Evaluate model
        loss, accuracy = evaluate_model(test_data=test_data, model=model)

        # Plot model
        model_name = create_new_dir(k, epochs)
        draw_learning_curves(model.history, model_name)

        # Export model
        export_model(
            model, 
            export_dir='model_files/'+model_name +'/', 
            label_filename=str(k + "_labels"), 
            overwrite=True,
            export_format=ExportFormat.SAVED_MODEL,
            save_model_filename=str(k + "_model_saved")
            )

        print("done done")
        #quit()
