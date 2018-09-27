if __name__ == '__main__':
    from keras import backend as K
    import numpy as np
    from OnlineAugmentation import ImageDataGenerator
    import cnn

    images = np.load('X.npy')
    labels = np.load('Y.npy')

    image_rows, image_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        images = images.reshape(images.shape[0], 3, image_rows, image_cols)
        input_shape = (3, image_rows, image_cols)
    else:
        images = images.reshape(images.shape[0], image_rows, image_cols, 3)
        input_shape = (image_rows, image_cols, 3)


    tiny_cnn = cnn.Tiny(input_shape, class_count = 2)

    trainer = cnn.Trainer()
    trainer.input = images
    trainer.output = labels
    trainer.data_generator = ImageDataGenerator(
        rotation_range = 90,
        horizontal_flip = True,
        vertical_flip = True,
        data_format = K.image_data_format(),
        blurring = False,
        white_to_color = True
    )
    trainer.train(tiny_cnn.model)

    tiny_cnn.save_to_file('models/tiny_cnn.h5')
