try:
    import configparser as ConfigParser
except ImportError:
    import ConfigParser

import keras
from keras import backend as K

def parseTrainingOptions(configFile):
    settings = {}
    config = ConfigParser.RawConfigParser(allow_no_value = True)
    config.read(configFile)
    settings['model_type'] = config.get("train", "model_type")
    settings['activation'] = config.get("train", "activation")
    settings['loss'] = config.get("train", "loss")
    settings['lr'] = float(config.get("train", "lr"))
    settings['decay'] = float(config.get("train", "decay"))
    settings['momentum'] = float(config.get("train", "momentum"))
    settings['nesterov'] = config.get("train", "nesterov")

    settings['batch_size'] = int(config.get("train", "batch_size"))
    settings['epochs'] = int(config.get("train", "epochs"))
    settings['verbose'] = int(config.get("train", "verbose"))
    return settings

class Camnet():
    config_file_path=''

    def __init__(self, config_file_path='./config.cfg'):
        self.config_file_path=config_file_path

    def get_Model(self, config_file_path):
        settings = parseTrainingOptions(config_file_path)

        if settings['model_type']=='resnet':
            base_model = keras.applications.resnet50.ResNet50(
                                                              include_top=True,
                                                              weights='imagenet'
                                                              )
            finetuning = keras.layers.Dense(1, activation='sigmoid', name='predictions')(base_model.layers[-2].output)
            model = keras.models.Model(input=base_model.input, output=finetuning)
        elif settings['model_type']=='resnet101':
            model = resnet101_model(settings, 3, 1)
        else:
            print('Initializing default model...')
            model = resnet101_model(settings, 3, 1)
        opt = keras.optimizers.SGD(
                                   lr=settings['lr'],
                                   momentum=settings['momentum'],
                                   decay=settings['decay'],
                                   nesterov=settings['nesterov']
                                   )

        model.compile(
                      loss=settings['loss'],
                      optimizer=opt,
                      metrics=['accuracy']
                      )
        return model
