from __future__ import absolute_import
import sys
#sys.path.append('./camnet/')
#sys.path.append('./scripts/keras_vis_rcv/vis/utils')
sys.path.append('./scripts/')
import stain_tools
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import greycoprops, greycomatrix
import cv2
from xml.etree.ElementTree import parse
#from os import listdir
#from os.path import join, isfile, exists, splitext
import skimage.segmentation
import skimage.filters
import skimage.morphology
import scipy.stats

from keras import backend as K

from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.backprop_modifiers import get
from vis.utils import utils

import sklearn.model_selection
import sklearn.linear_model

import math

def get_normalizer():
    normalizer = stain_tools.ReinhardNormalizer()
    # note:
    # normalizing_patch is a patch from camelyon17
    # used as a reference for the normalizer.
    # To apply to a different data
    # change normalizing_patch with a 224x224
    # patch from  the dataset currently used
    patch = np.load('data/normalizing_patch.npy')
    normalizer.fit(patch)
    return normalizer

def normalize_patch(patch, normalizer):
    return np.float64(normalizer.transform(patch))

def load_patches_db(path='./datasets/training/'):
    masks = []
    stats = []
    nuclei_list = []
    files = os.listdir(path)
    patches = np.load(os.path.join(path, files[0]))
    #print files[0]
    #print len(patches)
    images = np.zeros((len(files)*len(patches),224,224,3))
    c = 0
    for f in files:
        patches = np.load(os.path.join(path, f))
        for patch in patches:
            image, mask, nuclei, stat = patch
            images[c] = image[:,:,:3]
            masks.append(mask)
            stats.append(stat)
            nuclei_list.append(nuclei)
            c += 1
    return images, masks, nuclei_list, stats

def get_norm_patches(path='./datasets/training/'):
    patches, masks, nuclei, stats = load_patches_db(path=path)
    normalizer = get_normalizer()
    normalized_patches = [normalize_patch(np.uint8(x), normalizer) for x in patches]
    return np.array(normalized_patches),masks,  nuclei, stats

def avg_patch_area(patch_statistics):
    summ = 0.0
    for s in patch_statistics:
        summ += s.area
    return summ/len(patch_statistics)
def avg_area(stats):
    avg_areas = []
    patch_statistics = stats
    for st in (patch_statistics):
            avg_areas.append(avg_patch_area(st))
    return avg_areas

def max_area_in_patch(patch_statistics):
    areas=[]
    for s in patch_statistics:
        areas.append(s.area)
    return max(np.array(areas))
def max_area(stats):
    max_areas = []
    patch_statistics = stats
    for st in (patch_statistics):
            max_areas.append(max_area_in_patch(st))
    return max_areas
def max_diameter_in_patch(patch_statistics):
    diameters=[]
    for s in patch_statistics:
        diameters.append(s.equivalent_diameter)
    return max(np.array(diameters))
def max_diameter(stats):
    max_diameters=[]
    patch_statistics = stats
    for st in (patch_statistics):
            max_diameters.append(max_diameter_in_patch(st))
    return max_diameters

def max_mja_in_patch(patch_statistics):
    mja=[]
    for s in patch_statistics:
        mja.append(s.major_axis_length)
    return np.mean(np.array(mja))

def max_mjal(stats):
    max_mjals=[]
    patch_statistics = stats
    for st in (patch_statistics):
            max_mjals.append(max_mja_in_patch(st))
    return max_mjals
def centroids_in_patch(patch_statistics):
    centroids=[]
    for s in patch_statistics:
        centroids.append(s.centroid)
    return centroids
def nuclei_centroid(stats):
    nuclei_centroids=[]
    patch_statistics = stats
    for st in (patch_statistics):
            nuclei_centroids.append(centroids_in_patch(st))
    return nuclei_centroids
def cxarea_in_patch(patch_statistics):
    cxarea=[]
    for s in patch_statistics:
        cxarea.append(s.convex_area)
    return np.sum(np.asarray(cxarea, dtype=np.float32))/len(cxarea)
def convex_area(stats):
    convex_areas=[]
    patch_statistics = stats
    for st in (patch_statistics):
            convex_areas.append(cxarea_in_patch(st))
    return convex_areas
def eccentricity_in_patch(patch_statistics):
    ecc=[]
    for s in patch_statistics:
        ecc.append(s.eccentricity)
    return np.sum(np.asarray(ecc, dtype=np.float32))/len(ecc)
def eccentricity(stats):
    eccentricitys=[]
    patch_statistics = stats
    for st in (patch_statistics):
            eccentricitys.append(eccentricity_in_patch(st))
    return eccentricitys
def perimeter_in_patch(patch_statistics):
    per=[]
    for s in patch_statistics:
        per.append(s.perimeter)
    return np.sum(np.asarray(per, dtype=np.float32))/len(patch_statistics)
def perimeter(stats):
    perimeters=[]
    patch_statistics = stats
    for st in (patch_statistics):
            perimeters.append(perimeter_in_patch(st))
    return perimeters
def orientation_in_patch(patch_statistics):
    per=[]
    for s in patch_statistics:
        per.append(s.orientation)
    return per
def orientation(stats):
    orientations=[]
    patch_statistics = stats
    for st in (patch_statistics):
            orientations.append(orientation_in_patch(st))
    return orientations
def solidity_in_patch(patch_statistics):
    per=[]
    for s in patch_statistics:
        per.append(s.solidity)
    return np.sum(np.asarray(per, dtype=np.float32))/len(patch_statistics)
def solidity(stats):
    soliditys=[]
    patch_statistics = stats
    for st in (patch_statistics):
            soliditys.append(solidity_in_patch(st))
    return soliditys
def intensity_in_patch(patch_statistics):
    per=[]
    for s in patch_statistics:
        per.append(s.euler_number)
    return np.sum(np.asarray(per, dtype=np.float32))/len(patch_statistics)
def intensity(stats):
    soliditys=[]
    patch_statistics = stats
    for st in (patch_statistics):
            soliditys.append(intensity_in_patch(st))
    return soliditys
def nuclei_morphology(stats):
    nuclei_feats = {}
    #nuclei_feats['max_areas'] = max_area(stats)
    nuclei_feats['area'] = avg_area(stats)
    #nuclei_feats['max_diams'] = max_diameter(stats)
    #nuclei_feats['max_mjals'] = max_mjal(stats)
    nuclei_feats['mjaxis'] = max_mjal(stats)
    #nuclei_feats['nuclei_centroids'] = nuclei_centroid(stats)
    #nuclei_feats['convex_areas'] = convex_area(stats)
    nuclei_feats['eccentricity'] = eccentricity(stats)
    nuclei_feats['perimeter'] = perimeter(stats)
    #nuclei_feats['orientations'] = orientation(stats)
    nuclei_feats['euler'] = intensity(stats)
    return nuclei_feats

def nuclei_texture(patches, nuclei):
    texture_feats={}
    texture_feats['contrast'] = []
    texture_feats['dissimilarity'] = []
    #texture_feats['homogeneity'] = []
    texture_feats['ASM'] = []
    #texture_feats['energy']= []
    texture_feats['correlation'] = []
    for p in range(len(patches)):
        gray_patch=cv2.cvtColor(np.uint8(patches[p]), cv2.COLOR_BGR2GRAY)
        copy = nuclei[p].copy()
        copy/=255.0
        copy[nuclei[p]==0] = -1
        isolated_grayscale_nuclei = gray_patch*nuclei[p]/255.0
        isolated_grayscale_nuclei[copy==-1]=257
        patch = np.asarray(isolated_grayscale_nuclei, dtype=np.int)
        glcm = greycomatrix(patch, [5], [0], 258, symmetric=True, normed=True)
        glcm=glcm[:256,:256]
        texture_feats['contrast'].append(greycoprops(glcm, 'contrast')[0, 0])
        texture_feats['dissimilarity'].append(greycoprops(glcm, 'dissimilarity')[0, 0])
        #texture_feats['homogeneity'].append(greycoprops(glcm, 'homogeneity')[0, 0])
        texture_feats['ASM'].append(greycoprops(glcm, 'ASM')[0, 0])
        #texture_feats['energy'].append(greycoprops(glcm, 'energy')[0, 0])
        texture_feats['correlation'].append(greycoprops(glcm, 'correlation')[0, 0])
    return texture_feats

def compute_tcav_with_losses(input_tensor, losses, seed_input, wrt_tensor=None, grad_modifier='absolute'):
    """
    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.

        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
            ### NB. Here we can introduce our fl(x). The gradients will be computed wrt that tensor.

        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')

    Returns:
        The normalized gradients of `seed_input` with respect to weighted `losses`.
        ### NB. Here you will have to add the dot product with the normalized direction
            of the concept vector.
    """
    print 'wrt_tensor', wrt_tensor

    from keras.layers import Reshape
    #wrt_tensor = Reshape((14,14,1024,))(wrt_tensor)
    #print 'wrt_tensor', wrt_tensor
    #return

    opt = Optimizer(input_tensor, losses, wrt_tensor=wrt_tensor, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)[1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    #grads = np.max(grads, axis=channel_idx)
    return utils.normalize(grads)[0]


def compute_tcav(model, layer_idx, filter_indices, seed_input,
                       wrt_tensor=None, backprop_modifier=None, grad_modifier='absolute'):
    """Computes a Conceptual Sensitivity score `.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        seed_input: The model input for which activation map needs to be visualized.

        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
            ### NB. This will become the output of the
                layer at which Sensitivity is computed

        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')

    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        Not sure yet.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return compute_tcav_with_losses(model.input, losses, seed_input, wrt_tensor, grad_modifier)

##- -- patch extraction

def get_vertex(i):
    """Return a list of int coordinates of the nuclei annotation """
    nuclei_contour = []
    coordinates = i.getchildren()
    for data_point in coordinates:
        nuclei_contour.append(np.array([int(np.float32(data_point.attrib['X'])), int(np.float32(data_point.attrib['Y']))]))
    return nuclei_contour
def draw_contours(mask_image, point):
    for i,npary in enumerate(point):
            li_xy = npary.flatten()
            d_y, d_x = li_xy[::2],li_xy[1::2]
            mask_image[d_x,d_y] = 255.0
    return mask_image

def draw_cells(mask_image,nuclei_contour):
    #mask=np.zeros(mask_image[...,0].shape,np.uint8)
    #con=cv2.drawContours(mask, nuclei_contour,-1,(255,0,0), 2)
    #tum =cv2.drawContours(tum_im, tumor_contours,-1,(255,0,0), 3)
    #annotations_mask=cv2.fillPoly(mask_image, pts =[cn for cn in nuclei_contour], color=(255,255,255))
    annotations_mask=cv2.fillPoly(mask_image, pts =np.array([[[cn[0],cn[1]] for cn in nuclei_contour]], dtype=np.int32), color=(255,255,255))
    return annotations_mask

def draw_mask(mask_image,file_name):
        plt.figure(figsize=(30,30))
        plt.imshow(mask_image)
        plt.savefig('./breast_nuclei/Annotations/masks/'+file_name+'.png')
def get_mask(treeIterator, file_name):
    mask_image_cells=np.zeros((1001,1001))
    mask_image=np.zeros((1001,1001))
    for i in treeIterator:
        if i.tag =='Vertices':
            point_coordinates = get_vertex(i)
            mask_image = draw_contours(mask_image, point_coordinates)
            cells = draw_cells(mask_image_cells, point_coordinates)
    #mask_dir='./breast_nuclei/Annotations/masks'
    #np.save(mask_dir+'/'+file_name, mask_image)
    #draw_mask(mask_image, file_name)
    return mask_image, mask_image_cells
def sample_location(mode='random', width=1000, height=1000, verbose=0):
    return  np.random.randint(0, width-224), np.random.randint(0,height-224)
def cut_patch(slide, mask, mask_cells, patch_size=224, mode = 'random'):
    x, y = sample_location()
    #print x, y
    patch_image=np.asarray(slide.read_region((x,y),0,(patch_size,patch_size)))
    patch_annotations=mask[y:y+224, x:x+224]
    patch_nuclei = mask_cells[y:y+224, x:x+224]
    patch_nuclei = skimage.segmentation.clear_border(patch_nuclei)
    return patch_image, patch_annotations, patch_nuclei

def visualize_overlap(patch, annotations):
    plt.figure()
    plt.imshow(patch)
    plt.imshow(annotations, alpha=0.5)
    plt.savefig('./training/'+str(np.random.random_integers(0,100000))+'.png')
    plt.close()

def get_patch_statistics(patch, annotations):
    """Returns a set of average measurements of
        nuclei in the patch """
    thr = skimage.filters.threshold_otsu(annotations)
    annotations = skimage.morphology.closing(annotations>thr, skimage.morphology.square(3))
    clear_annotations = skimage.segmentation.clear_border(annotations)
    #clear_annotations = skimage.morphology.dilation(clear_annotations)
    clear_annotations = clear_annotations * 1
    #visualize_overlap(patch, clear_annotations)
    label_annotations = skimage.morphology.label(clear_annotations)
    measure_properties = skimage.measure.regionprops(label_annotations)
    return measure_properties, clear_annotations
def get_n_patches(slide, mask_image,mask_image_cells, n = 1):
    patches_set = []
    for i in range(n):
        patch, anno, nuclei=cut_patch(slide, mask_image, mask_image_cells)
        #visualize_overlap(patch, anno)
        measures, clear_annotations = get_patch_statistics(patch, anno)
        patches_set.append([patch, clear_annotations, nuclei, measures])
    return patches_set

def get_cv_training_set(imgs_dir, repetition):
    training_set = []
    print imgs_dir, os.listdir(imgs_dir)
    breast_wsis = os.listdir(imgs_dir)
    print 'breast_wsis', breast_wsis
    new_dir = './datasets/training/'+str(repetition)+'/'
    os.mkdir(new_dir)
    for file_name in breast_wsis:
        print file_name
        slide = OpenSlide(imgs_dir+file_name)
        tree = parse('./datasets/breast_nuclei/Annotations/'+file_name[:-4]+'.xml')
        root=tree.getroot()
        treeIterator = root.getiterator()
        mask_image, mask_image_cells = get_mask(treeIterator, file_name)
        patches_set = get_n_patches(slide, mask_image, mask_image_cells, n = 50)
        np.save(new_dir+file_name[:-4], patches_set)

        training_set.append(patches_set)
    return training_set
#===

def corr_analysis(feature, pred):
    return scipy.stats.pearsonr(np.array(feature), np.array(pred))


class Loss(object):
    """Abstract class for defining the loss function to be minimized.
    The loss function should be built by defining `build_loss` function.

    The attribute `name` should be defined to identify loss function with verbose outputs.
    Defaults to 'Unnamed Loss' if not overridden.
    """
    def __init__(self):
        self.name = "Unnamed Loss"

    def __str__(self):
        return self.name

    def build_loss(self):
        """Implement this function to build the loss function expression.
        Any additional arguments required to build this loss function may be passed in via `__init__`.

        Ideally, the function expression must be compatible with all keras backends and `channels_first` or
        `channels_last` image_data_format(s). `utils.slicer` can be used to define data format agnostic slices.
        (just define it in `channels_first` format, it will automatically shuffle indices for tensorflow
        which uses `channels_last` format).

        ```python
        # theano slice
        conv_layer[:, filter_idx, ...]

        # TF slice
        conv_layer[..., filter_idx]

        # Backend agnostic slice
        conv_layer[utils.slicer[:, filter_idx, ...]]
        ```

        [utils.get_img_shape](vis.utils.utils.md#get_img_shape) is another optional utility that make this easier.

        Returns:
            The loss expression.
        """
        raise NotImplementedError()


class ActivationMaximization(Loss):
    """A loss function that maximizes the activation of a set of filters within a particular layer.

    Typically this loss is used to ask the reverse question - What kind of input image would increase the networks
    confidence, for say, dog class. This helps determine what the network might be internalizing as being the 'dog'
    image space.

    One might also use this to generate an input image that maximizes both 'dog' and 'human' outputs on the final
    `keras.layers.Dense` layer.
    """
    def __init__(self, layer, filter_indices):
        """
        Args:
            layer: The keras layer whose filters need to be maximized. This can either be a convolutional layer
                or a dense layer.
            filter_indices: filter indices within the layer to be maximized.
                For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

                If you are optimizing final `keras.layers.Dense` layer to maximize class output, you tend to get
                better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
                output can be maximized by minimizing scores for other classes.
        """
        super(ActivationMaximization, self).__init__()
        self.name = "ActivationMax Loss"
        self.layer = layer
        self.filter_indices = utils.listify(filter_indices)

    def build_loss(self):
        layer_output = self.layer.output

        # For all other layers it is 4
        is_dense = K.ndim(layer_output) == 2

        loss = 0.
        for idx in self.filter_indices:
            if is_dense:
                loss += -K.mean(layer_output[:, idx])
            else:
                # slicer is used to deal with `channels_first` or `channels_last` image data formats
                # without the ugly conditional statements.
                loss += -K.mean(layer_output[utils.slicer[:, idx, ...]])

        return loss
### regression
def solve_regression(inputs, y, n_splits=3, n_repeats=1, random_state=12883823, verbose=1):
    scores=[]
    max_score = 0
    direction = None
    dirs=[]
    rkf = sklearn.model_selection.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    counter = 0
    for train, test in rkf.split(inputs):
        if verbose:
            print 'N. ', counter, '..'
            print len(inputs[train])
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(inputs[train], y[train])
        trial_score = reg.score(inputs[test], y[test])
        dirs.append(reg.coef_)
        #print 'y[train]', y[train]
        scores.append(trial_score)
        if trial_score > max_score:
            direction = reg.coef_
        if verbose:
            print trial_score
        counter += 1
    if verbose:
        print np.mean(scores)
        i=0
        while i+1<len(dirs):
            print 'angle: ', py_ang(dirs[i], dirs[i+1])
            i+=1
    return np.mean(scores), direction
def py_ang(v1, v2):
    cos = np.dot(v1,v2)
    return np.arccos(cos/(np.linalg.norm(v1) * np.linalg.norm(v2)))

def plot_scores(scores, legend, legend_entry, color):
    import matplotlib
    mu = np.mean(scores)
    variance = np.std(scores)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,matplotlib.mlab.normpdf(x, mu, sigma), color=color)
    plt.scatter([mu,mu,mu,mu], [0,0.25,0.5,0.75],marker='*',c=color, s=3)
    legend.append(legend_entry)
    return mu, variance, sigma, legend
