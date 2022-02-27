from scipy import ndimage
import numpy as np

def largest_connected_region(image):
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    s = ndimage.generate_binary_structure(3,1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output 