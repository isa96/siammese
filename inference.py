import os
from utils import (
    get_siamese_network, 
    SiameseModel, 
    extract_encoder,
    classify_images, 
    read_image,
    )


def run(folder = "test_image"):
    """
    ref:
        - ignored weight not used, https://github.com/tensorflow/tensorflow/issues/43554
    """
    siamese_network = get_siamese_network()
    siamese_model = SiameseModel(siamese_network)

    # load weight of siammese
    siamese_model.load_weights("model_siammese/siamese_model-final").expect_partial()

    encoder = extract_encoder(siamese_model)

    # load encoder model
    encoder.load_weights("model_siammese/encoder").expect_partial()

    list_image = [os.path.join(folder, filename) for filename in os.listdir(folder)]

    # NOTE: iterate N^2 of list_image to define optimize threshold of distance algorithm
    for _i, i in enumerate(list_image):
        i = read_image(i)
        for _j, j in enumerate(list_image):
            j = read_image(j)
            proba, distance = classify_images(i, j, encoder)
            result = "Similar" if proba == 0 else "Different"

            print(f"[LOGS] : {_i} - {_j} - {distance} -> {result}")


if __name__ == "__main__":
    run()