import numpy as np
import pandas as pd
from pathlib import Path
from typical_cells import (
    get_h_e_ims,
    center_ims,
    top_S_bot_E,
    subsample_dict,
    h_comp_to_rgb,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bokeh.models import Slider, ColumnDataSource, Button
from PIL import Image
from bokeh.plotting import figure, show
from bokeh.layouts import column, layout, row
from bokeh.io import curdoc
from umap import UMAP

res_path = Path(r"D:\Results\TMA_results\test_run23\node_preds")


# load the images and scores from the .npz file
with np.load(res_path.parent / "sarc_images.npz") as f:
    sarc_images = f["images"]
    props_S = {"scores": f["scores"], "areas": f["areas"], "circ": f["circ"]}
with np.load(res_path.parent / "ep_images.npz") as f:
    ep_images = f["images"]
    props_E = {"scores": f["scores"], "areas": f["areas"], "circ": f["circ"]}

# get top 10% of sarc images and bottom 10% of ep images
sarc_images = sarc_images[: int(0.1 * len(sarc_images)), :]
ep_images = ep_images[-int(0.1 * len(ep_images)) :, :]

props_S, props_E = top_S_bot_E(props_S, props_E, frac=0.1)

n_comp = 16
im_size = 40
# subsample images so that we can do PCA on them
subsamp = 5
ep_images = ep_images[::subsamp, :]
props_E = subsample_dict(props_E, subsamp)
subsamp = int(subsamp / 4)
sarc_images = sarc_images[::subsamp, :]
props_S = subsample_dict(props_S, subsamp)

sarc_images_h, ep_images_h, sarc_images_e, ep_images_e = get_h_e_ims(
    sarc_images, ep_images
)

sarc_images_c, s_mean1, s_mean2 = center_ims(sarc_images_h)
ep_images_c, e_mean1, e_mean2 = center_ims(ep_images_h)

# find top 5 principal components of E and S images
pca_E = PCA(n_components=n_comp)  # , kernel='cosine')
# pca_E.fit(np.reshape(ep_images, (ep_images.shape[0], -1)))
pca_E.fit(ep_images_c)
pca_S = PCA(n_components=n_comp)  # , kernel='cosine')
# pca_S.fit(np.reshape(sarc_images, (sarc_images.shape[0], -1)))
pca_S.fit(sarc_images_c)

components_E = pca_E.transform(ep_images_c)
components_S = pca_S.transform(sarc_images_c)
components_E = StandardScaler().fit_transform(components_E)
components_S = StandardScaler().fit_transform(components_S)

# find principal components of the principal components
pca_E2 = PCA(n_components=5)
pca_E2.fit(components_E)
pca_S2 = PCA(n_components=5)
pca_S2.fit(components_S)

umap_E = UMAP(n_components=2, metric="cosine")
umap_S = UMAP(n_components=2, metric="cosine")
# umap_E.fit(components_E)
# umap_S.fit(components_S)
umap_E.fit(ep_images_c)
umap_S.fit(sarc_images_c)

# make scatter plots of the first two principal components
scat_E = figure(width=400, height=400, tools=["tap", "wheel_zoom", "box_select", "pan"])
# pca_E2_ds = ColumnDataSource(data={'x': pca_E2.transform(components_E)[:,0], 'y': pca_E2.transform(components_E)[:,1]})
pca_E2_ds = ColumnDataSource(
    data={"x": umap_E.embedding_[:, 0], "y": umap_E.embedding_[:, 1]}
)
scat_E.circle(x="x", y="y", source=pca_E2_ds)

scat_S = figure(width=400, tools=["tap", "wheel_zoom", "box_select", "pan"], height=400)
# pca_S2_ds = ColumnDataSource(data={'x': pca_S2.transform(components_S)[:,0], 'y': pca_S2.transform(components_S)[:,1]})
pca_S2_ds = ColumnDataSource(
    data={"x": umap_S.embedding_[:, 0], "y": umap_S.embedding_[:, 1]}
)
scat_S.circle(x="x", y="y", source=pca_S2_ds)

comp_weights_E = np.zeros(n_comp)
comp_weights_S = np.zeros(n_comp)


def make_comp_slider(name):
    return Slider(
        start=-105, end=105, value=0, step=0.1, name=name, title=name, width=250
    )


def comp_slider_cb_E(obj, attr, old, new):
    # change corresponding pca component weight and then rebuild weighted image
    comp_weights_E[int(obj.name.split("_")[1])] = new
    im = image_from_comps(pca_E, comp_weights_E, im_size, e_mean1)
    print(im.shape)
    # import pdb; pdb.set_trace()
    source_E.data["image"] = [im]


def comp_slider_cb_S(obj, attr, old, new):
    # change corresponding pca component weight and then rebuild weighted image
    comp_weights_S[int(obj.name.split("_")[1])] = new
    im = image_from_comps(pca_S, comp_weights_S, im_size, s_mean1)
    print(im.shape)
    source_S.data["image"] = [im]


def bind_cb_obj(cb_obj, cb):
    def wrapped(attr, old, new):
        cb(cb_obj, attr, old, new)

    return wrapped


def image_as_uint32(im):
    img = np.empty(im.shape[:2], dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((*im.shape[:2], 4))
    view[:] = im
    return img


def image_from_comps(pca_obj, comps, im_size, mean):
    # given vector in PCA space, and PCA object, return an image
    # built from the weighted sum of the PCA components
    im = np.zeros((im_size, im_size))
    im += np.reshape(mean, (im_size, im_size))
    for i, comp in enumerate(comps):
        im += comp * np.reshape(pca_obj.components_[i], (im_size, im_size))
    # get image as uint32
    im = image_as_uint32(
        np.concatenate(
            (
                h_comp_to_rgb(im, mean_e=e_mean1),
                255 * np.ones((im_size, im_size, 1), dtype=np.uint8),
            ),
            axis=2,
        )
    )
    return im


im = np.zeros((im_size, im_size, 4), dtype=np.uint8)

im = image_as_uint32(im)
print(im.shape)
pE = figure(x_range=(0, 10), y_range=(0, 10), width=400, height=400)
source_E = ColumnDataSource({"image": [im]})
pE.image_rgba(image="image", x=0, y=0, dw=10, dh=10, source=source_E)

pS = figure(x_range=(0, 10), y_range=(0, 10), width=400, height=400)
source_S = ColumnDataSource({"image": [im]})
pS.image_rgba(image="image", x=0, y=0, dw=10, dh=10, source=source_S)

ep_column = [make_comp_slider(f"ep_{i}") for i in range(n_comp)]
# bind callbacks to sliders
for slider in ep_column:
    slider.on_change("value", bind_cb_obj(slider, comp_slider_cb_E))
# add a button to set sliders to components of a random E cell
def random_E_cell_cb(attr):
    # random cell from ep_images_h
    rand_im = ep_images_h[np.random.randint(0, ep_images_h.shape[0]), :, :]
    comps = pca_E.transform(np.reshape(rand_im, (1, -1)))
    # set sliders to components of random cell
    # import pdb; pdb.set_trace()
    for i, slider in enumerate(ep_column):
        slider.value = comps[0, i]
    source_E_real.data["image"] = [
        image_as_uint32(
            np.concatenate(
                (
                    h_comp_to_rgb(rand_im),
                    255 * np.ones((im_size, im_size, 1), dtype=np.uint8),
                ),
                axis=2,
            )
        )
    ]


random_E_cell = Button(label="Random E Cell", button_type="success")
random_E_cell.on_click(random_E_cell_cb)

ep_sliders = column(ep_column)

sarc_column = [make_comp_slider(f"sarc_{i}") for i in range(n_comp)]
# bind callbacks to sliders
for slider in sarc_column:
    slider.on_change("value", bind_cb_obj(slider, comp_slider_cb_S))
# add a button to set sliders to components of a random E cell
def random_S_cell_cb(attr):
    # random cell from ep_images_h
    rand_ind = np.random.randint(0, sarc_images_h.shape[0])
    rand_im = sarc_images_h[rand_ind, :, :]
    comps = pca_S.transform(np.reshape(rand_im, (1, -1)))
    # set sliders to components of random cell
    for i, slider in enumerate(sarc_column):
        slider.value = comps[0, i]
    # rand_im = np.squeeze(sarc_images[rand_ind,16:-16,16:-16,:])
    rand_im = h_comp_to_rgb(rand_im)
    source_S_real.data["image"] = [
        image_as_uint32(
            np.concatenate(
                (rand_im, 255 * np.ones((im_size, im_size, 1), dtype=np.uint8)), axis=2
            )
        )
    ]


random_S_cell = Button(label="Random S Cell", button_type="success")
random_S_cell.on_click(random_S_cell_cb)

sarc_sliders = column(sarc_column)

# button to zero all component sliders
zero_button = Button(label="Zero All", button_type="success")


def zero_cb(attr):
    for slider in ep_column:
        slider.value = 0
    for slider in sarc_column:
        slider.value = 0


zero_button.on_click(zero_cb)


def scatter_select_cb_E(attr, old, new):
    """when a cell is selected in the scatter plot, calculate the pca components
    and set the sliders to those values
    """
    # get the pca components of the selected cell
    comps = pca_E.transform(np.reshape(ep_images_h[new, :, :], (len(new), -1)))
    comps = np.mean(comps, keepdims=True, axis=0)
    # set sliders to components of selected cell
    for i, slider in enumerate(ep_column):
        slider.value = comps[0, i]
    # set the image to the selected cell
    source_E_real.data["image"] = [
        image_as_uint32(
            np.concatenate(
                (
                    h_comp_to_rgb(
                        np.mean(ep_images_h[new, :, :], keepdims=True, axis=0)
                    ),
                    255 * np.ones((im_size, im_size, 1), dtype=np.uint8),
                ),
                axis=2,
            )
        )
    ]


def scatter_select_cb_S(attr, old, new):
    """when a cell is selected in the scatter plot, calculate the pca components
    and set the sliders to those values
    """
    print(new)
    # get the pca components of the selected cell
    comps = pca_S.transform(np.reshape(sarc_images_h[new, :, :], (len(new), -1)))
    comps = np.mean(comps, keepdims=True, axis=0)
    # set sliders to components of selected cell
    for i, slider in enumerate(sarc_column):
        slider.value = comps[0, i]
    # set the image to the selected cell
    source_S_real.data["image"] = [
        image_as_uint32(
            np.concatenate(
                (
                    h_comp_to_rgb(
                        np.mean(sarc_images_h[new, :, :], keepdims=True, axis=0)
                    ),
                    255 * np.ones((im_size, im_size, 1), dtype=np.uint8),
                ),
                axis=2,
            )
        )
    ]


pca_E2_ds.selected.on_change("indices", scatter_select_cb_E)
pca_S2_ds.selected.on_change("indices", scatter_select_cb_S)


# 2 more plots for actual ims if select random cells
pE_real = figure(x_range=(0, 10), y_range=(0, 10), width=400, height=400)
source_E_real = ColumnDataSource({"image": [im]})
pE_real.image_rgba(image="image", x=0, y=0, dw=10, dh=10, source=source_E_real)

pS_real = figure(x_range=(0, 10), y_range=(0, 10), width=400, height=400)
source_S_real = ColumnDataSource({"image": [im]})
pS_real.image_rgba(image="image", x=0, y=0, dw=10, dh=10, source=source_S_real)

l = layout(
    [
        [
            [pE, pE_real],
            [pS, pS_real],
            [ep_sliders, random_E_cell],
            [sarc_sliders, random_S_cell, zero_button],
            [scat_E, scat_S],
        ]
    ]
)
curdoc().add_root(l)
