from pathlib import Path
from tiatoolbox.annotation.storage import SQLiteStore
from shapely.affinity import translate
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA, MiniBatchDictionaryLearning
import numpy as np
from PIL import Image
from math import acos, degrees
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.color import rgb2hed, hed2rgb
import cv2
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (40, 40)
n_edge = 16
white_hed = rgb2hed(np.array((255,255,255)).reshape(1,1,3))
mpl.use('agg')

"""Take the most epithelioid and most sarcomatoid cells from epithelioid and
sarcomatoid examples respectively, align their images,
and then calculate the eigenvectors o the images to show typical morpology 
of the cell types.
"""
def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()

def h_comp_to_rgb(h, save_path=None, mean_e=None):
    #import pdb; pdb.set_trace()
    # invert image if border is darker than bulk
    #if np.mean(h[0] + h[-1]) > np.mean(h):
        #h = -h
    h = h - np.min(h)
    h = h/np.max(h)
    h = np.reshape((255*h).astype(np.uint8), (72-2*n_edge,72-2*n_edge,1))
    hed = np.zeros((h.shape[0], h.shape[1], 3))
    #if np.mean(h[0,:] + h[-1,:]) > 128:
        #h = 255 - h
    # create rgb using matplotlib colormap
    cmap = cm.get_cmap('Blues')
    rgb = (255*cmap(np.squeeze(h)/255)).astype(np.uint8)

    #hed[:] = white_hed
    #hed[:,:,0] = np.squeeze(h)
    #if mean_e is not None:
        #hed[:,:,1] = np.reshape(mean_e, (72-2*n_edge,72-2*n_edge))
    #rgb = (255*hed2rgb(hed/255)/np.max(hed2rgb(hed/255))).astype(np.uint8)
    #if np.mean(rgb) < 128:
        #rgb = 255 - rgb
    if save_path is not None:
        Image.fromarray((255*rgb).astype(np.uint8)).save(save_path)
    return rgb[:,:,:3]

def center_ims(ims):
    n_samples, *n_features = ims.shape
    ims = ims.reshape(n_samples, -1)
    mean1 = ims.mean(axis=0)
    # Global centering (focus on one feature, centering all samples)
    ims_c = ims - mean1

    # Local centering (focus on one sample, centering all features)
    mean2 = ims_c.mean(axis=1)[:,None]
    ims_c -= mean2
    return ims_c, mean1, mean2

def subsample_dict(dict, n):
    #subsample each array in dict by taking every nth value
    for key in dict:
        dict[key] = dict[key][::n]
    return dict
    

def plot_typical_images(sarc_images, ep_images, props_S, props_E, clust='all'):
    """Given a group of sarcomatoid and a group of epithelioid images,
    plot typical images for each group calulated ina  variety of ways.
    """
    n_comp = 15
    save_path = res_path.parent / f'typical_cells_{clust}'
    save_path.mkdir(exist_ok=True)
    #show naive mean E and S images
    mean_E_image = np.mean(ep_images, axis=0)
    mean_S_image = np.mean(sarc_images, axis=0)
    plt.subplot(1,2,1)
    im = h_comp_to_rgb(mean_E_image, save_path / 'mean_E_image.png')
    plt.imshow(im)
    plt.title('naive mean E')
    plt.subplot(1,2,2)
    im = h_comp_to_rgb(mean_S_image, save_path / 'mean_S_image.png')
    plt.imshow(im)
    plt.title('naive mean S')
    plt.show()

    sarc_images_c, _, _ = center_ims(sarc_images)
    ep_images_c, _, _ = center_ims(ep_images)

    #find top 5 principal components of E and S images
    pca_E = PCA(n_components=n_comp)#, kernel='cosine')
    #pca_E.fit(np.reshape(ep_images, (ep_images.shape[0], -1)))
    pca_E.fit(ep_images_c)
    pca_S = PCA(n_components=n_comp)#, kernel='cosine')
    #pca_S.fit(np.reshape(sarc_images, (sarc_images.shape[0], -1)))
    pca_S.fit(sarc_images_c)

    def show_comps(decomp_E, decomp_S, decomp_name='comp'):
        #show top 5 principal components of E and S images
        for i in range(n_comp):
            #comp_img_E = np.reshape((255*decomp_E.components_[i,:]/np.max(decomp_E.components_[i,:])).astype(np.uint8), (72-2*n_edge,72-2*n_edge,1))
            #img_E_rgb = np.zeros((72-2*n_edge,72-2*n_edge,3), dtype=np.uint8)
            #img_E_rgb[:,:,0] = np.squeeze(comp_img_E)
            plt.subplot(1,2,1)
            plt.imshow(h_comp_to_rgb(decomp_E.components_[i,:], save_path / f'E_{decomp_name}_{i}.png'))#, cmap='gray')
            plt.title('E PC ' + str(i))
            #comp_img_S = np.reshape((255*decomp_S.components_[i,:]/np.max(decomp_S.components_[i,:])).astype(np.uint8), (72-2*n_edge,72-2*n_edge,1))
            #img_S_rgb = np.zeros((72-2*n_edge,72-2*n_edge,3), dtype=np.uint8)
            #img_S_rgb[:,:,0] = np.squeeze(comp_img_S)
            plt.subplot(1,2,2)
            plt.imshow(h_comp_to_rgb(decomp_S.components_[i,:], save_path / f'S_{decomp_name}_{i}.png'))#, cmap='gray')
            plt.title('S PC ' + str(i))
            plt.show()

    show_comps(pca_E, pca_S, 'PCA')

    #find top 5 NMF components of E and S images
    pca_E = NMF(n_components=n_comp)#, kernel='cosine')
    #pca_E.fit(np.reshape(ep_images, (ep_images.shape[0], -1)))
    pca_E.fit(ep_images.reshape(ep_images.shape[0], -1))
    pca_S = NMF(n_components=n_comp)#, kernel='cosine')
    #pca_S.fit(np.reshape(sarc_images, (sarc_images.shape[0], -1)))
    pca_S.fit(sarc_images.reshape(sarc_images.shape[0], -1))

    show_comps(pca_E, pca_S, 'NMF')

    pca_E = FastICA(
        n_components=n_comp, max_iter=400, whiten="arbitrary-variance", tol=15e-5
    )
    pca_E.fit(ep_images_c)
    pca_S = FastICA(
        n_components=n_comp, max_iter=400, whiten="arbitrary-variance", tol=15e-5
    )
    pca_S.fit(sarc_images_c)

    show_comps(pca_E, pca_S, 'ICA')

    #plot principal component i against principal component j for stacked E and S images
    i,j=0,1
    plt.scatter(pca_E.transform(np.reshape(np.vstack((ep_images, sarc_images)), (ep_images.shape[0]+sarc_images.shape[0], -1))).T[i], pca_E.transform(np.reshape(np.vstack((ep_images, sarc_images)), (ep_images.shape[0]+sarc_images.shape[0], -1))).T[j], c=np.vstack((props_E['scores'], props_S['scores'])), cmap='Reds')
    plt.title('E PC ' + str(i) + ' vs. E PC ' + str(j))
    plt.show()

    do_umap = False
    if do_umap:
         #perform a umap reduction of the images
        umap_E = UMAP(n_neighbors=10, n_components=2, metric='cosine')
        umap_E.fit(np.reshape(ep_images, (ep_images.shape[0], -1)))
        umap_S = UMAP(n_neighbors=10, n_components=2, metric='cosine')
        umap_S.fit(np.reshape(sarc_images, (sarc_images.shape[0], -1)))

        #show umap embedding of E and S images
        plt.subplot(1,2,1)
        plt.scatter(umap_E.embedding_[:,0], umap_E.embedding_[:,1], c=scores_E)
        plt.title('E umap')
        plt.subplot(1,2,2)
        plt.scatter(umap_S.embedding_[:,0], umap_S.embedding_[:,1], c=scores_S)
        plt.title('S umap')
        plt.show()

        #cluster the embedded images using KMeans
        kmeans_E = KMeans(n_clusters=5, random_state=0)
        kmeans_E.fit(umap_E.embedding_)
        kmeans_S = KMeans(n_clusters=5, random_state=0)
        kmeans_S.fit(umap_S.embedding_)

        #calulate the mean image for each cluster
        cluster_ind_E = kmeans_E.predict(umap_E.embedding_)
        cluster_ind_S = kmeans_S.predict(umap_S.embedding_)
    else:
        n_clust=10
        #cluster the images PCA decompositions using KMeans
        kmeans_E = AgglomerativeClustering(n_clusters=n_clust, affinity='cosine', linkage='average')
        kmeans_E.fit(pca_E.transform(np.reshape(ep_images, (ep_images.shape[0], -1))))
        kmeans_S = AgglomerativeClustering(n_clusters=n_clust, affinity='cosine', linkage='average')
        kmeans_S.fit(pca_S.transform(np.reshape(sarc_images, (sarc_images.shape[0], -1))))

        #calulate the mean image for each cluster
        cluster_ind_E = kmeans_E.labels_# kmeans_E.predict(np.reshape(ep_images, (ep_images.shape[0], -1)))
        cluster_ind_S = kmeans_S.labels_#kmeans_S.predict(np.reshape(sarc_images, (sarc_images.shape[0], -1)))
    
    cluster_images_E = []
    cluster_images_S = []
    E_sizes = []
    S_sizes = []
    for i in range(n_clust):
        e_size = np.sum(cluster_ind_E==i)
        s_size = np.sum(cluster_ind_S==i)
        E_sizes.append(e_size)
        S_sizes.append(s_size)
        print(f'cluster {i} has sizes {e_size}, {s_size}')
        cluster_images_E.append(np.mean(ep_images[cluster_ind_E == i, :], axis=0))
        cluster_images_S.append(np.mean(sarc_images[cluster_ind_S == i, :], axis=0))
    
    #sort clusters by size
    sorted_E = np.argsort(E_sizes)[::-1]
    sorted_S = np.argsort(S_sizes)[::-1]

    #show the mean images for each cluster
    for i in range(n_clust):
        im_E = h_comp_to_rgb(cluster_images_E[sorted_E[i]], save_path / f'cluster_E_{i}.png')
        plt.subplot(1,2,1)
        plt.imshow(im_E)#, cmap='gray')
        plt.title('E PC ' + str(i))
        im_S = h_comp_to_rgb(cluster_images_S[sorted_S[i]], save_path / f'cluster_S_{i}.png')
        plt.subplot(1,2,2)
        plt.imshow(im_S)#, cmap='gray')
        plt.title('S PC ' + str(i))
        plt.show()


def get_h_e_ims(sarc_images, ep_images):
    # remove n_edge pixels from all borders of the image to avoid edge effects
    sarc_images = sarc_images[:,n_edge:-n_edge,n_edge:-n_edge,:]
    ep_images = ep_images[:,n_edge:-n_edge,n_edge:-n_edge,:]

    if False:
        #make a circular mask which is zero for all pixels more than
        # a distance of 11 from the center
        mask = np.zeros(sarc_images.shape[1:], dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i+0.5-mask.shape[0]/2)**2 + (j+0.5-mask.shape[1]/2)**2 > 11**2:
                    mask[i,j,:] = 255

        #apply the mask to all images
        sarc_images = sarc_images + mask[None,:,:,:]
        ep_images = ep_images + mask[None,:,:,:]
        #sarc_images[sarc_images==0] = 255
        #ep_images[ep_images==0] = 255

    #convert to grayscale (naive way)
    #sarc_images = np.mean(sarc_images, axis=3)
    #ep_images = np.mean(ep_images, axis=3)

    #convert to h&e
    sarc_images_h = (255*rgb2hed(sarc_images)[:,:,:,0]).astype(np.uint8)
    ep_images_h = (255*rgb2hed(ep_images)[:,:,:,0]).astype(np.uint8)

    sarc_images_e = (255*rgb2hed(sarc_images)[:,:,:,1].astype(np.uint8))
    ep_images_e = (255*rgb2hed(ep_images)[:,:,:,1].astype(np.uint8))
    return sarc_images_h, ep_images_h, sarc_images_e, ep_images_e

def top_S_bot_E(props_S, props_E, frac=0.1):
    #get the top frac of sarcomatoid images and bottom frac of epithelioid properties
    #and return the filtered props_S and props_E dicts
    n_S = int(frac*len(props_S['scores']))
    n_E = int(frac*len(props_E['scores']))
    props_S['scores'] = props_S['scores'][:n_S]
    props_S['areas'] = props_S['areas'][:n_S]
    props_S['circ'] = props_S['circ'][:n_S]
    props_E['scores'] = props_E['scores'][-n_E:]
    props_E['areas'] = props_E['areas'][-n_E:]
    props_E['circ'] = props_E['circ'][-n_E:]
    return props_S, props_E

if __name__ == '__main__':
    #dets_path = Path(r'D:\Mesobank_TMA\mesobank_proj\detections\stores')
    #patch_path = Path(r'D:\Mesobank_TMA\mesobank_proj\det_patches')
    dets_path = Path(r'D:\QuPath_Projects\Meso_TMA\detections\stores')
    patch_path = Path(r'D:\QuPath_Projects\Meso_TMA\det_patches')
    res_path = Path(r'D:\Results\TMA_results\test_run23\node_preds')

    dets_list = list(dets_path.glob('*.db'))

    if False:
        # get the images of each cell
        ep_images, sarc_images = [],[]
        scores_E, scores_S = [], []
        areas_E, areas_S = [], []
        circ_E, circ_S = [], []
        for core in dets_list:
            print(f'processing core {core.stem}')
            with open(patch_path/f'{core.stem}.npy', 'rb') as f:
                patches = np.load(f) 
            core_db = SQLiteStore(core)
            labels = [core.stem.split('_')[1]]*len(patches)
            node_scores = pd.read_csv(res_path/(f'GNN_scores_{core.stem}.csv'))
            node_scores['score'] = (1 + node_scores['score_S'].values - node_scores['score_E'].values)/2
            rotated_images= []
            for i, ann in enumerate(core_db.values()):
                #print(f'point at {ann.geometry.centroid}')
                #scores.append([ann.properties['scoreE'], ann.properties['scoreS']])
                geom = translate(ann.geometry, -ann.geometry.centroid.x, -ann.geometry.centroid.y)
                pca = PCA(n_components=1)
                pca.fit(np.array(geom.exterior.coords))
                comp1 = pca.components_[0,:]
                theta = acos(comp1[1])   # angle between vector [0,1] and principal component
                im = patches[i, :]
                if i%100 == -1:
                    plt.subplot(1,2,1)
                    plt.imshow(im)
                    plt.scatter(np.array(geom.exterior.coords)[:,0]+36.5, np.array(geom.exterior.coords)[:,1]+36.5)
                #mask out the pixels in im that aren't inside geometry using cv2
                mask = np.zeros(im.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, (np.array(geom.buffer(2.0).exterior.coords)+36.5).astype(np.int32), 1)
                #im = im*mask[:,:,None]
                im[mask!=1, :] = 0 # what is best?
                
                rot_angle = degrees(theta) if comp1[0] < 0 else -degrees(theta)
                #print(f'rotating by {rot_angle} degs')
                im_rot = Image.fromarray(im).rotate(rot_angle, fillcolor='white')
                rotated_images.append(np.array(im_rot))
                if i%100 == -1:
                    plt.subplot(1,2,2)
                    plt.imshow(im_rot)
                    plt.show()
                if labels[0] == 'E':
                    areas_E.append(ann.properties['Nucleus: Area Âµm^2'])
                    circ_E.append(ann.properties['Nucleus: Circularity'])
                elif labels[0] == 'S':
                    areas_S.append(ann.properties['Nucleus: Area Âµm^2'])
                    circ_S.append(ann.properties['Nucleus: Circularity'])
            rotated_images = np.array(rotated_images)
            if labels[0] == 'E':
                ep_images.append(rotated_images)
                scores_E.append(node_scores['score'].values[:-1,None])
            elif labels[0] == 'S':
                sarc_images.append(rotated_images)
                scores_S.append(node_scores['score'].values[:-1,None])
            
        sarc_images = np.vstack(sarc_images)
        ep_images = np.vstack(ep_images)
        scores_S = np.vstack(scores_S)
        scores_E = np.vstack(scores_E)
        areas_S = np.array(areas_S)
        areas_E = np.array(areas_E)
        circ_S = np.array(circ_S)
        circ_E = np.array(circ_E)

        #sort the images and scores by their 'sarcomatoidness' score
        sarc_images = sarc_images[np.argsort(scores_S,0)[::-1],:]
        ep_images = ep_images[np.argsort(scores_E,0)[::-1],:]
        scores_S = scores_S[np.argsort(scores_S,0)[::-1],0]
        scores_E = scores_E[np.argsort(scores_E,0)[::-1],0]
        areas_S = areas_S[np.argsort(scores_S,0)[::-1]]
        areas_E = areas_E[np.argsort(scores_E,0)[::-1]]
        circ_S = circ_S[np.argsort(scores_S,0)[::-1]]
        circ_E = circ_E[np.argsort(scores_E,0)[::-1]]

        #save the images and scores to an .npz file
        np.savez(res_path.parent/'sarc_images_b.npz', images=np.squeeze(sarc_images), scores=scores_S, areas=areas_S, circ=circ_S)
        np.savez(res_path.parent/'ep_images_b.npz', images=np.squeeze(ep_images), scores=scores_E, areas=areas_E, circ=circ_E)
    else:
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

    def index_all_in_dict(d, idx):
        #index all the values in a dict with a list of indices
        #and return as a new dict
        new_d = {}
        for k,v in d.items():
            new_d[k] = v[idx]
        return new_d


    #show typical images for all sarcomatoid and all epithelioid images
    #plot_typical_images(sarc_images, ep_images, scores_S, scores_E, subsamp=40)

    #show typical images for top 10% of sarcomation images and bottom 10% of epithelioid images
    #plot_typical_images(sarc_images[:int(0.1*len(sarc_images)),:], ep_images[-int(0.1*len(ep_images)):,:], scores_S[:int(0.1*len(scores_S))], scores_E[-int(0.1*len(scores_E)):], subsamp=4)

    #filter out top and bottom perc% by area as probably outliers
    perc=5
    limits = np.percentile(props_S['areas'], [perc,100-perc])
    sarc_images = sarc_images[np.squeeze(np.logical_and(props_S['areas']>limits[0], props_S['areas']<limits[1])),:]
    props_S = index_all_in_dict(props_S, np.where(np.logical_and(props_S['areas']>limits[0], props_S['areas']<limits[1]))[0])
    limits = np.percentile(props_E['areas'], [perc,100-perc])
    ep_images = ep_images[np.squeeze(np.logical_and(props_E['areas']>limits[0], props_E['areas']<limits[1])),:]
    props_E = index_all_in_dict(props_E, np.where(np.logical_and(props_E['areas']>limits[0], props_E['areas']<limits[1]))[0])

    sarc_images_h, ep_images_h, sarc_images_e, ep_images_e = get_h_e_ims(sarc_images, ep_images)

    #scatter plot of circularity vs area for sarcomatoid and epithelioid images
    plt.subplot(1,2,1)
    plt.scatter(props_S['circ'], props_S['areas'], label='sarcomatoid')
    plt.subplot(1,2,2)
    plt.scatter(props_E['circ'], props_E['areas'], label='epithelioid')
    plt.show()

    #histogram plot of area for sarcomatoid and epithelioid images
    plt.subplot(1,2,1)
    plt.hist(props_S['areas'], bins=100, label='sarcomatoid')
    plt.subplot(1,2,2)
    plt.hist(props_E['areas'], bins=100, label='epithelioid')

    #cluster the sarcomatoid images in area-circularity space
    #and plot the clusters
    X = np.hstack((props_S['circ'], props_S['areas']))
    kmeans_S = KMeans(n_clusters=4, random_state=0).fit(StandardScaler().fit_transform(X))
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=kmeans_S.labels_, cmap='viridis')
    plt.title('Sarcomatoid')
    plt.xlabel('Circularity')
    plt.ylabel('Area')
    #repeat for epithelioid images
    plt.subplot(1,2,2)
    X = np.hstack((props_E['circ'], props_E['areas']))
    kmeans_E = KMeans(n_clusters=4, random_state=0).fit(StandardScaler().fit_transform(X))
    plt.scatter(X[:,0], X[:,1], c=kmeans_E.labels_, cmap='viridis')
    plt.title('Epithelioid')
    plt.xlabel('Circularity')
    plt.ylabel('Area')
    plt.show()

    #plot typical images for all
    plot_typical_images(sarc_images_h, ep_images_h, props_S, props_E, 'all')
    #plot_typical_images(sarc_images_e[:int(0.1*len(sarc_images)),:], ep_images_e[-int(0.1*len(ep_images)):,:], scores_S[:int(0.1*len(scores_S))], scores_E[-int(0.1*len(scores_E)):], subsamp=4)
    #for each cluster, plot the typical images
    for i in range(4):
        print('Plotting typical images for cluster %d' % i)
        idx_S = kmeans_S.labels_==i
        idx_E = kmeans_E.labels_==i
        print(f'Cluster sizes are S: {np.sum(idx_S)} and E: {np.sum(idx_E)}')
        plot_typical_images(sarc_images_h[idx_S,:], ep_images_h[idx_E,:], index_all_in_dict(props_S, idx_S), index_all_in_dict(props_E, idx_E), f'cluster_{i}')


    #do lda discriminating between top 10% sarc images and bottom 10% ep images
    lda = LinearDiscriminantAnalysis(n_components=1)
    top_bot_images = np.vstack((sarc_images_h[:int(0.1*len(sarc_images_h)),:], ep_images_h[-int(0.1*len(ep_images_h)):,:]))
    top_bot_labels = np.hstack((np.ones(int(0.1*len(sarc_images_h))), np.zeros(int(0.1*len(ep_images_h)))))
    top_bot_images,_,_ = center_ims(top_bot_images)
    lda.fit(top_bot_images, top_bot_labels)

    #show coefs
    #lda_im = np.zeros((72-2*n_edge,72-2*n_edge,3), dtype=np.uint8)
    #lda_im[:,:,0] = np.squeeze(255*lda.coef_[0,:].reshape(sarc_images_h.shape[1:])/np.max(lda.coef_[0,:]))
    plt.imshow(h_comp_to_rgb(lda.coef_[0,:]))
    plt.show()
    print('done')












