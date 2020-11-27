import numpy as np
import matplotlib.pyplot as plt

def plot_place_field(place_field):
    """ Plot place field with added color bar."

        Args:
            place_field - place field as np.array
    """
    if place_field.ndim == 1:
        im = np.expand_dims(place_field, 1)   
        im = np.repeat(im, 3, 1).T
        plt.yticks([])
        plt.xticks(np.arange(place_field.shape[0]))
    else:
        im = place_field
        plt.yticks(np.arange(place_field.shape[0]))
        plt.xticks(np.arange(place_field.shape[1]))

    plt.imshow(im)
    plt.colorbar()
    plt.show()

"""
# quick test
# TODO unit tests

plot_place_field(np.array([0,0,0,1,2,3,2,1,0,0,0,0,0,0]))
plot_place_field(np.array([[0,0,0,1,2,3,2,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,2,3,2,1,0,0,0]]))
"""
