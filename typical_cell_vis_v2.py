import numpy as np
import pandas as pd
from pathlib import Path
from typical_cells import get_h_e_ims, center_ims, top_S_bot_E, subsample_dict, h_comp_to_rgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bokeh.models import Slider, ColumnDataSource, Button, MultiChoice, LinearColorMapper
from PIL import Image
from bokeh.plotting import figure, show
from bokeh.layouts import column, layout, row
from bokeh.io import curdoc
from bokeh.transform import linear_cmap
from umap import UMAP
from sklearn.neighbors import KernelDensity

res_path = Path(r'D:\Results\TMA_results\test_run23\node_preds')
n_comp = 16
im_size = 40

#load the images and scores from the .npz file
with np.load(res_path.parent/'sarc_images.npz') as f:
    sarc_images = f['images']
    props_S = {'scores': f['scores'], 'areas': f['areas'], 'circ': f['circ']}
with np.load(res_path.parent/'ep_images.npz') as f:
    ep_images = f['images']
    props_E = {'scores': f['scores'], 'areas': f['areas'], 'circ': f['circ']}

#get top 10% of sarc images and bottom 10% of ep images
sarc_images = sarc_images[:int(0.1*len(sarc_images)),:]
ep_images = ep_images[-int(0.1*len(ep_images)):,:]

props_S, props_E = top_S_bot_E(props_S, props_E, frac=0.1)

#subsample images so that we can do PCA on them
subsamp=5
ep_images = ep_images[::subsamp,:]
props_E = subsample_dict(props_E, subsamp)
subsamp = int(subsamp/4)
sarc_images = sarc_images[::subsamp,:]
props_S = subsample_dict(props_S, subsamp)

all_images = np.concatenate((sarc_images, ep_images), axis=0)
#take only the central 40x40 pixels of the 72x72 images
all_images = all_images[:,16:56,16:56,:]
#replace white background with gray
sarc_images[sarc_images==255] = 50

sarc_images_h, ep_images_h, sarc_images_e, ep_images_e = get_h_e_ims(sarc_images, ep_images)
all_images_h = np.concatenate((sarc_images_h, ep_images_h), axis=0)

sarc_images_c, s_mean1, s_mean2 = center_ims(sarc_images_h)
ep_images_c, e_mean1, e_mean2 = center_ims(ep_images_h)
all_images_c, a_mean1, a_mean2 = center_ims(np.concatenate((sarc_images_h, ep_images_h), axis=0))
all_images_orig_c, ao_mean1, ao_mean2 = center_ims(np.concatenate((sarc_images, ep_images), axis=0))

#find top 5 principal components of E and S images
pca_E = PCA(n_components=n_comp)#, kernel='cosine')
#pca_E.fit(np.reshape(ep_images, (ep_images.shape[0], -1)))
pca_E.fit(ep_images_c)
pca_S = PCA(n_components=n_comp)#, kernel='cosine')
#pca_S.fit(np.reshape(sarc_images, (sarc_images.shape[0], -1)))
pca_S.fit(sarc_images_c)
pca_A = PCA(n_components=n_comp)
pca_A.fit(all_images_c)

components_E = pca_E.transform(ep_images_c)
components_S = pca_S.transform(sarc_images_c)
components_A = pca_A.transform(all_images_c)
components_E = StandardScaler().fit_transform(components_E)
components_S = StandardScaler().fit_transform(components_S)
components_A = StandardScaler().fit_transform(components_A)
components_A_S = components_A[:len(components_S),:]
components_A_E = components_A[-len(components_E):,:]

# find principal components of the principal components
pca_E2 = PCA(n_components=5)
pca_E2.fit(components_E)
pca_S2 = PCA(n_components=5)
pca_S2.fit(components_S)

umap_E = UMAP(n_components=2, metric='cosine')
umap_S = UMAP(n_components=2, metric='cosine')
umap_A = UMAP(n_components=2, metric='cosine')
umap_AC = UMAP(n_components=2, metric='cosine')
umap_orig = UMAP(n_components=2, metric='minkowski', metric_kwds={"p": 3}, n_neighbors=10, densmap=False, dens_lambda=0.2, repulsion_strength=1.0, n_epochs=500, target_weight=0.4)
#umap_E.fit(components_E)
#umap_S.fit(components_S)
umap_E.fit(ep_images_c)
umap_S.fit(sarc_images_c)
umap_A.fit(all_images_c)
umap_AC.fit(components_A)
#umap_orig.fit(all_images_orig_c, y=np.concatenate((np.ones(len(components_S)), np.zeros(len(components_E))), axis=0))
umap_orig.fit(np.reshape(all_images_orig_c,(all_images.shape[0],-1)), y=np.concatenate((np.ones(len(components_S)), np.zeros(len(components_E))), axis=0))

#make scatter plots of the first two principal components
scat_E = figure(width=400, height=400, tools=["tap", "wheel_zoom", "box_select", "pan"])
#pca_E2_ds = ColumnDataSource(data={'x': pca_E2.transform(components_E)[:,0], 'y': pca_E2.transform(components_E)[:,1]})
pca_E2_ds = ColumnDataSource(data={'x': umap_E.embedding_[:,0], 'y': umap_E.embedding_[:,1]})
scat_E.circle(x='x', y='y', source=pca_E2_ds)

scat_S = figure(width=400, tools=["tap", "wheel_zoom", "box_select", "pan"], height=400)
#pca_S2_ds = ColumnDataSource(data={'x': pca_S2.transform(components_S)[:,0], 'y': pca_S2.transform(components_S)[:,1]})
pca_S2_ds = ColumnDataSource(data={'x': umap_S.embedding_[:,0], 'y': umap_S.embedding_[:,1]})
scat_S.circle(x='x', y='y', source=pca_S2_ds)

scat_A = figure(width=500, tools=["tap", "wheel_zoom", "box_select", "lasso_select", "pan"], height=500)
#pca_S2_ds = ColumnDataSource(data={'x': pca_S2.transform(components_S)[:,0], 'y': pca_S2.transform(components_S)[:,1]})

pca_A2_ds = ColumnDataSource(data={'x': umap_orig.embedding_[:,0], 'y': umap_orig.embedding_[:,1], 'score': np.vstack((props_S['scores'], props_E['scores']))})
scat_A.circle(x='x', y='y', fill_color=linear_cmap('score', 'Turbo256', 0, 1), line_color=linear_cmap('score', 'Turbo256', 0, 1), fill_alpha=0.5, line_alpha=0.5, source=pca_A2_ds)

comp_weights_E = np.zeros(n_comp)
comp_weights_S = np.zeros(n_comp)
comp_weights_A = np.zeros(n_comp)

def mk_density(cell_scores, x):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(cell_scores.reshape(-1, 1))
    #print(kde.score_samples(x.reshape(-1, 1)))
    #import pdb; pdb.set_trace()
    dens = np.exp(kde.score_samples(x.reshape(-1, 1)))
    return dens

def make_comp_slider(name):
    return Slider(start=-105, end=105, value=0, step=0.1, name=name, title=name, width=250)

def comp_slider_cb_E(obj, attr, old, new):
    #change corresponding pca component weight and then rebuild weighted image
    comp_weights_E[int(obj.name.split('_')[1])] = new
    im = image_from_comps(pca_E, comp_weights_E, im_size, e_mean1)
    print(im.shape)
    #import pdb; pdb.set_trace()
    source_E.data['image'] = [im]

def comp_slider_cb_S(obj, attr, old, new):
    #change corresponding pca component weight and then rebuild weighted image
    comp_weights_S[int(obj.name.split('_')[1])] = new
    im = image_from_comps(pca_S, comp_weights_S, im_size, s_mean1)
    print(im.shape)
    source_S.data['image'] = [im]

def comp_slider_cb_A(obj, attr, old, new):
    #change corresponding pca component weight and then rebuild weighted image
    comp_weights_A[int(obj.name.split('_')[1])] = new
    im = image_from_comps(pca_A, comp_weights_A, im_size, a_mean1)
    print(im.shape)
    source_A.data['image'] = [im]

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
    #given vector in PCA space, and PCA object, return an image
    #built from the weighted sum of the PCA components
    im = np.zeros((im_size, im_size))
    im += np.reshape(mean, (im_size, im_size))
    for i, comp in enumerate(comps):
        im += comp*np.reshape(pca_obj.components_[i],(im_size, im_size))
    #get image as uint32
    im = image_as_uint32(np.concatenate((h_comp_to_rgb(im, mean_e=e_mean1), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))
    return im

im = np.zeros((im_size, im_size, 4), dtype=np.uint8)

im = image_as_uint32(im)
print(im.shape)
pE = figure(x_range=(0,10), y_range=(0,10), width=400, height=400)
source_E = ColumnDataSource({'image': [im]})
pE.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_E)

pS = figure(x_range=(0,10), y_range=(0,10), width=400, height=400)
source_S = ColumnDataSource({'image': [im]})
pS.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_S)

pA = figure(x_range=(0,10), y_range=(0,10), width=400, height=400)
source_A = ColumnDataSource({'image': [im]})
pA.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_A)

ep_column = [make_comp_slider(f'ep_{i}') for i in range(n_comp)]
#bind callbacks to sliders
for slider in ep_column:
    slider.on_change('value', bind_cb_obj(slider, comp_slider_cb_E))
#add a button to set sliders to components of a random E cell
def random_E_cell_cb(attr):
    #random cell from ep_images_h
    rand_im = ep_images_h[np.random.randint(0, ep_images_h.shape[0]),:,:]
    comps = pca_E.transform(np.reshape(rand_im,(1,-1)))
    #set sliders to components of random cell
    #import pdb; pdb.set_trace()
    for i, slider in enumerate(ep_column):
        slider.value = comps[0,i]
    source_E_real.data['image'] = [image_as_uint32(np.concatenate((h_comp_to_rgb(rand_im), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]

random_E_cell = Button(label='Random E Cell', button_type='success')
random_E_cell.on_click(random_E_cell_cb)

ep_sliders = column(ep_column)

sarc_column = [make_comp_slider(f'sarc_{i}') for i in range(n_comp)]
#bind callbacks to sliders
for slider in sarc_column:
    slider.on_change('value', bind_cb_obj(slider, comp_slider_cb_S))

all_column = [make_comp_slider(f'all_{i}') for i in range(n_comp)]
#bind callbacks to sliders
for slider in all_column:
    slider.on_change('value', bind_cb_obj(slider, comp_slider_cb_A))
all_sliders = column(all_column)

#add a button to set sliders to components of a random E cell
def random_S_cell_cb(attr):
    #random cell from ep_images_h
    rand_ind = np.random.randint(0, sarc_images_h.shape[0])
    rand_im = sarc_images_h[rand_ind,:,:]
    comps = pca_S.transform(np.reshape(rand_im,(1,-1)))
    #set sliders to components of random cell
    for i, slider in enumerate(sarc_column):
        slider.value = comps[0,i]
    #rand_im = np.squeeze(sarc_images[rand_ind,16:-16,16:-16,:])
    rand_im = h_comp_to_rgb(rand_im)
    source_S_real.data['image'] = [image_as_uint32(np.concatenate((rand_im, 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]

random_S_cell = Button(label='Random S Cell', button_type='success')
random_S_cell.on_click(random_S_cell_cb)

sarc_sliders = column(sarc_column)

#button to zero all component sliders
zero_button = Button(label='Zero All', button_type='success')
def zero_cb(attr):
    for slider in ep_column:
        slider.value = 0
    for slider in sarc_column:
        slider.value = 0
    for slider in all_column:
        slider.value = 0
zero_button.on_click(zero_cb)

def scatter_select_cb_E(attr, old, new):
    """when a cell is selected in the scatter plot, calculate the pca components
    and set the sliders to those values
    """
    #get the pca components of the selected cell
    comps = pca_E.transform(np.reshape(ep_images_h[new,:,:],(len(new),-1)))
    comps = np.mean(comps, keepdims=True, axis=0)
    #set sliders to components of selected cell
    for i, slider in enumerate(ep_column):
        slider.value = comps[0,i]
    #set the image to the selected cell
    source_E_real.data['image'] = [image_as_uint32(np.concatenate((h_comp_to_rgb(np.mean(ep_images_h[new,:,:], keepdims=True, axis=0)), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]

def scatter_select_cb_S(attr, old, new):
    """when a cell is selected in the scatter plot, calculate the pca components
    and set the sliders to those values
    """
    print(new)
    #get the pca components of the selected cell
    comps = pca_S.transform(np.reshape(sarc_images_h[new,:,:],(len(new),-1)))
    comps = np.mean(comps, keepdims=True, axis=0)
    #set sliders to components of selected cell
    for i, slider in enumerate(sarc_column):
        slider.value = comps[0,i]
    #set the image to the selected cell
    source_S_real.data['image'] = [image_as_uint32(np.concatenate((h_comp_to_rgb(np.mean(sarc_images_h[new,:,:], keepdims=True,axis=0)), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]

def scatter_select_cb_A(attr, old, new):
    """when a cell is selected in the scatter plot, calculate the pca components
    and set the sliders to those values
    """
    print(new)
    #get the pca components of the selected cell
    comps = pca_A.transform(np.reshape(all_images_h[new,:,:],(len(new),-1)))
    comps = np.mean(comps, keepdims=True, axis=0)
    #set sliders to components of selected cell
    for i, slider in enumerate(all_column):
        slider.value = comps[0,i]
    #set the image to the selected cell
    source_A_real.data['image'] = [image_as_uint32(np.concatenate((np.mean(all_images[new,:,:,:], keepdims=False,axis=0), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]
    #source_A_real.data['image'] = [image_as_uint32(np.concatenate((h_comp_to_rgb(np.mean(all_images_h[new,:,:], keepdims=True,axis=0)), 255*np.ones((im_size, im_size, 1), dtype=np.uint8)),axis=2))]

ep_dens_ds = ColumnDataSource({'x': np.linspace(-10,10,100), 'y': np.zeros((100,1))})
sarc_dens_ds = ColumnDataSource({'x': np.linspace(-10,10,100), 'y': np.zeros((100,1))})

#make a new plot to show the densities of the cell groups
dens_plot = figure(plot_width=500, plot_height=300, title='Densities')
dens_plot.line('x', 'y', source=ep_dens_ds, color='blue', line_width=2, legend_label='ep')
dens_plot.line('x', 'y', source=sarc_dens_ds, color='red', line_width=2, legend_label='sarc')

def pc_select_cb(attr, old, new):
    """when a principal component is selected, calculate a density
    for both sarc_images_c and ep_images_c, and put them in the corresponding
    datasource
    """
    if len(new) > 0:
        #get the selected component
        comp = int(new[0])
        #make the x-range to cover the entire range of both components_A_E and components_A_S
        x_range = np.linspace(np.min([np.min(components_A_E[:,comp]), np.min(components_A_S[:,comp])]), np.max([np.max(components_A_E[:,comp]), np.max(components_A_S[:,comp])]), 100)
        #get the density of the selected component for both sarc_images_c and ep_images_c
        ep_dens = mk_density(components_A_E[:,comp], x_range)
        sarc_dens = mk_density(components_A_S[:,comp], x_range)
        #set the data for the density plots
        ep_dens_ds.data['y'] = ep_dens
        sarc_dens_ds.data['y'] = sarc_dens
        ep_dens_ds.data['x'] = x_range
        sarc_dens_ds.data['x'] = x_range

#multichoice widget to select a principal component from a list
pc_selector = MultiChoice(title='Select a Principal Component', value=[], options=[str(i) for i in range(n_comp)], max_items = 1)
pc_selector.on_change('value', pc_select_cb)

pca_E2_ds.selected.on_change('indices', scatter_select_cb_E)
pca_S2_ds.selected.on_change('indices', scatter_select_cb_S)
pca_A2_ds.selected.on_change('indices', scatter_select_cb_A)


#2 more plots for actual ims if select random cells
pE_real = figure(x_range=(0,10), y_range=(0,10), width=400, height=400)
source_E_real = ColumnDataSource({'image': [im]})
pE_real.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_E_real)

pS_real = figure(x_range=(0,10), y_range=(0,10), width=400, height=400)
source_S_real = ColumnDataSource({'image': [im]})
pS_real.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_S_real)

pA_real = figure(x_range=(0,10), y_range=(0,10), width=500, height=500)
source_A_real = ColumnDataSource({'image': [im]})
pA_real.image_rgba(image='image', x=0, y=0, dw=10, dh=10, source=source_A_real)

#l = layout([[[pE,pE_real], [pS,pS_real], [ep_sliders, random_E_cell], [sarc_sliders, random_S_cell, zero_button],[scat_E, scat_S]]])
l = layout([[[pA_real], [all_sliders, zero_button], [scat_A, pc_selector, dens_plot]]])
curdoc().add_root(l)
