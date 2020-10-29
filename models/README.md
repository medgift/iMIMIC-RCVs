## Download the trained models 
The trained model weights can be downloaded at:

https://doi.org/10.5281/zenodo.4155010

How to load the models? 

    CONFIG_FILE='config.cfg'
    settings = functions.parseTrainingOptions(CONFIG_FILE)
    bc_model = models.getModel(settings)
    bc_model.load_weights('tumor_classifier.h5')

For more instructions check RCV_notebook.ipynb
