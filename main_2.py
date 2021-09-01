import tensorflow as tf
import matplotlib.pyplot as plt
import os

#from tensorflow_examples.lite.model_maker.core.optimization import warmup
#from tensorflow_examples.lite.model_maker.core.task.train_image_classifier_lib import HParams

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker.config import QuantizationConfig


import matplotlib.pyplot as plt
from datetime import datetime

assert tf.__version__.startswith('2')




# LOAD

#module_spec = hub.load_module_spec("./model_files/")
#height, width = hub.get_expected_image_size(module_spec)
#images = ...  # A batch of images with shape [batch_size, height, width, 3].
#module = hub.Module(module_spec)
#features = module(images)   # A batch with shape [batch_size, num_features].

def return_image_path_dict():
    return {
        #'image_path_IR'                         : '../saue bilder/Combined/IR_originale/',
        #'img_path_IR_duplicate_STRICT'          : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_STRICT/',
        #'img_path_IR_duplicate_BASIC'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_BASIC/',
        #'img_path_IR_duplicate_LOOSE'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_LOOSE/',
        #'img_path_IR_BLURRY_duplicate_STRICT'   : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_STRICT/',
        #'img_path_IR_BLURRY_duplicate_BASIC'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_BASIC/',
        #'img_path_IR_BLURRY_duplicate_LOOSE'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_LOOSE/',
        'img_path_VISUAL'                       : '../saue bilder/Combined/Visual/',
        #'img_path_VISUAL_duplicate_STRICT'      : '../saue bilder/Combined/Visual_removed_duplicate_STRICT/',
        #'img_path_VISUAL_duplicate_BASIC'       : '../saue bilder/Combined/Visual_removed_duplicate_BASIC/',
        #'img_path_VISUAL_duplicate_LOOSE'       : '../saue bilder/Combined/Visual_removed_duplicate_LOOSE/'
        }

def load_data_tflite_model_maker(path, split_trainrest_ratio=0.9, split_testval_ratio=0.5):
    data = image_classifier.DataLoader.from_folder(path, shuffle=True)
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

    # TODO: This is totally wrong. the variables are never changing.. wtf?!
    
    model = image_classifier.create(
        train_data=train_data,
        model_spec=model_spec,                  validation_data=validation_data,
        batch_size=batch_size,                  epochs=epochs,
        train_whole_model=train_whole_model,    dropout_rate=dropout_rate,              
        learning_rate=learning_rate,            momentum=momentum,                      
        shuffle=shuffle,                        use_augmentation=use_augmentation,      
        use_hub_library=use_hub_library,        warmup_steps=warmup_steps,              
        model_dir=model_dir,                    do_train=do_train
    )

    return model

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
    model.evaluate(test_data)

# plot classified images
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

# plot model
def draw_learning_curves(history, model_title):
    #plot_title = str(path.split("/")[-2])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    figure = plt.figure(figsize=(8, 8))
    plt.ion()
    plt.subplot(2, 1.1, 1.1)
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
    plt.close(figure)
    plt.savefig("plots/"+ model_title + ".png")
    #plt.show()

def get_local_time():
    d = datetime.now()
    return str(d.day) + "." + str(d.month) + " " + str(d.hour) + "." + str(d.minute)

def create_new_dir(model_name, epochs):
    local_time = get_local_time()
    new_dir = str(model_name) + "_" + str(epochs) + "_epochs_" + str(local_time) 
    parent_dir = "./model_files/"
    path = os.path.join(parent_dir, new_dir)
    os.mkdir(path)
    return path


if __name__ == '__main__':
    '''
    path_ir = '../saue bilder/Combined/IR_originale/'

    path = '../saue bilder/Combined/Visual_originale/'
    path_2 = '/mnt/GigaServer1/Haakosbo/bilder/Visual_originale/'
    path_3 = '../Visual_originale/'
    path_4 = '../saue bilder/Combined/Visual/'

    path = path
    '''


    for i, (k, v) in enumerate(return_image_path_dict().items()):
        
        path = v
        epochs = 10
        
        #dir_path = create_new_dir(model_name=k, epochs=epochs)
        
        split_trainrest_ratio = 0.8
        split_testval_ratio = 0.5

        # Load Data
        train_data, test_data, validation_data = load_data_tflite_model_maker(path, split_trainrest_ratio, split_testval_ratio)
        
        # Build model and initiate training
        model = image_classifier.create(
            train_data=train_data,                  
            model_spec='efficientnet_lite4', # 'efficientnet_lite0-4 'mobilenet_v2' 'resnet_50'                  
            validation_data=validation_data, # DEFAULT VALUES       
            batch_size=32,                 # 32
            epochs=epochs,                  # 5          
            dropout_rate=0.6,                # 0.2      
            learning_rate=0.002,            # 0.005
            momentum=0.9,                    # 0.9  
            shuffle=True,                    # False    
            use_augmentation=True,           # False 
            warmup_steps=None,               # None
            do_train=True,                  # True
            # For fine tuning                #
            train_whole_model=False,         # True    
            use_hub_library=False,            # if True then standard hub library Hparams are used.
            model_dir=".",                   
        )

        #model.trainable = True
        #print("layers in base model: ", len(model.module_layer))

        #fine_tune_at = 100

        #for layer in model.layers[:fine_tune_at]:
            #layer.trainable = False


        # TODO:
        # Create a way of fine tune the model prior to training. This might not be possible as
        # It could be problems with the TFLITE allowing "fine_tuneing" Aswell as tflite model are
        # created with transfer learning which implies that this training is fine tuning to begin with
        # model
        # If model_dir="correct path to the saved model" and use_hub_library=False we are doing fine_tuning
        # This could be very cool!
       
        # 


        
        
        # Evaluate model
        loss, accuracy = model.evaluate(test_data)
        #print("accuracy: " +str(accuracy))
        #print("loss: " + str(loss))


        # Plot model
        draw_learning_curves(model.history, k)

        
        #xx = model.train(train_data)
        #xx.history
        #print(xx.history)

        asd = model.predict_top_k(test_data)
        #print(asd)
        
        #plot classified pictures "Manual checking"
        #plot_classified_images(model=model, test_data=test_data)

        # post  quantization to shrink the size of the model
        #config = QuantizationConfig.for_float16()

        # Export model
        export_model(
            model, 
            export_dir='export_model/', 
            tflite_filename=str(k + "_fp16.tflite"),
            #quantization_config=config,
            #label_filename=str(k + "_labels"), 
            #overwrite=True,
            #export_format=ExportFormat.SAVED_MODEL,
            #save_model_filename=str(k + "_model_saved")
            )

            
        '''
        export_model(
            model, 
            export_dir='saved_model/', 
            label_filename=str(k + "_labels"), 
            overwrite=True,
            #export_format=ExportFormat.SAVED_MODEL,
            save_model_filename=str(k + "_model_saved")
        )
        '''
        #draw_learning_curves(model)

        print("done done")
        














    ''' GRAVEYARD 

    
            import tflite_model_maker as tflite
            from tflite_model_maker import model_spec
            from tflite_model_maker import image_classifier
            from tensorflow_examples.lite.model_maker.core.task import configs
            from tensorflow_examples.lite.model_maker.core import compat
            from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader 
            from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader
            from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
            from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import QuestionAnswerDataLoader
            from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import TextClassifierDataLoader
            from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
            from tensorflow_examples.lite.model_maker.core.task import configs
            from tensorflow_examples.lite.model_maker.core.task import image_classifier
            from tensorflow_examples.lite.model_maker.core.task import model_spec
            from tensorflow_examples.lite.model_maker.core.task import question_answer
            from tensorflow_examples.lite.model_maker.core.task import text_classifier

        model = image_classifier.create(
            train_data=train_data,                  
            model_spec='efficientnet_lite4', # 'efficientnet_lite0-4 'mobilenet_v2' 'resnet_50'                  
            validation_data=validation_data, # DEFAULT VALUES       
            batch_size=32,                 # 32
            epochs=100,                  # 5          
            dropout_rate=0.5,                # 0.2      
            learning_rate=0.0001,            # 0.005
            momentum=0.9,                    # 0.9  
            shuffle=True,                    # False    
            use_augmentation=True,           # False 
            warmup_steps=None,               # None
            do_train=True,                  # True
            # For fine tuning                #
            train_whole_model=False,         # True    
            use_hub_library=False,            # if True then standard hub library Hparams are used.
            model_dir='saved_model/saved_model/',                   
        )


        # TODO:
        # can  pass in do_train=False to build the model. then we can do a initial inference
        # then we can train and do inference to show the performance change. You must
        # probably change this method or actually just call it from here like a normal person..
        # Remember that we need to draw every graph accordingly check out tf_2 it has a way of adding
        # data to an ongoing plot.
        
        # initial performance
        loss, accuracy = model.evaluate(test_data)
        print("accuracy: " +str(accuracy))
        print("loss: " + str(loss))

        # Train model fine
        hparams_train = HParams(
            train_epochs=200, 
            do_fine_tuning=True, 
            batch_size=32,
            learning_rate=0.0001, 
            momentum=0.9, 
            dropout_rate=0.5, 
            l1_regularizer=0.0,
            l2_regularizer=0.0001, 
            label_smoothing=0.1, 
            validation_split=0.3, 
            do_data_augmentation=True, 
            rotation_range=360,          #rotation_range: Int. Degree range for random rotations.
            horizontal_flip=True,       #  see class ImageDataGenerator in
            width_shift_range=0.4,      #  tensorflow\python\keras\preprocessing\image.py
            height_shift_range=0.4, 
            shear_range=0.4,
            zoom_range=0.4,
            warmup_steps=None,
            model_dir="saved_model/saved_model/",
            )
        
        history_train_fine = model.train(train_data=train_data, validation_data=validation_data, hparams=hparams_train)

        # Evaluate model
        loss, accuracy = model.evaluate(test_data)
        print("accuracy: " +str(accuracy))
        print("loss: " + str(loss))
    
        model = image_classifier.create(
            train_data=train_data,
             model_spec='efficientnet_lite4',
             validation_data=validation_data,
             batch_size=32,
             epochs=100,
             train_whole_model=True,
             dropout_rate=0.5,
             learning_rate=0.0001,
             momentum=0.9,
             shuffle=True,
             use_augmentation=True,
             use_hub_library=False,
             warmup_steps=None,
             model_dir="model_files/",
             do_train=True
        )

        # fine tuning
        hparams_train = HParams(
            train_epochs=100, 
            do_fine_tuning=True, 
            batch_size=32,
            learning_rate=0.00001, 
            momentum=0.9, 
            dropout_rate=0.5, 
            l1_regularizer=0.0,
            l2_regularizer=0.0001, 
            label_smoothing=0.1, 
            validation_split=0.3, 
            do_data_augmentation=True, 
            rotation_range=360,          #rotation_range: Int. Degree range for random rotations.
            horizontal_flip=True,       #  see class ImageDataGenerator in
            width_shift_range=0.4,      #  tensorflow\python\keras\preprocessing\image.py
            height_shift_range=0.4, 
            shear_range=0.4,
            zoom_range=0.4,
            warmup_steps=None,
            model_dir="model_files/",
            )

        history_fine = model.train(train_data=train_data, validation_data=validation_data, hparams=hparams_train)




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