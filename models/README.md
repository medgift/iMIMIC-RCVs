The trained model weights can be downloaded at:

https://www.dropbox.com/sh/a56pyl4i8oj9zl9/AADR74UXUu_uZA6Omb3B9Q1Ea?dl=0

How to load the models? 

    CONFIG_FILE='./models/0528-1559/config.cfg'
    settings = functions.parseTrainingOptions(CONFIG_FILE)
    bc_model = models.getModel(settings)
    bc_model.load_weights('./models/0528-1559/tumor_classifier.h5')

For more instructions check RCV_notebook.ipynb