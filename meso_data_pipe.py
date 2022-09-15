import pandas as pd
from tiatoolbox.annotation.storage import SQLiteStore, Annotation
from pathlib import Path
import json
from shapely.geometry import Polygon
import math
from shapely.affinity import translate


"""processes raw detections from qupath and save them in an
AnnotationStore for use with MesoGraph"""

def to_core_space(core_db: SQLiteStore, top_left):
    #make annotation coords relative to core image top left
    core_db.translate_db(-top_left[0], -top_left[1])
    core_db.commit()

def mk_dbs_from_geojson(dataset = 'meso'):
    if dataset == 'meso':
        labels_path = Path(r'D:\Meso\core_labels_AIME.csv')
        dets_path = Path(r'D:\QuPath_Projects\Meso_TMA\detections')
        cents = pd.read_csv(r'D:\QuPath_Projects\Meso_TMA\core_cents.csv')
        um_per_pix=0.4415
        core_width = 2854
    else:
        labels_path = Path(r'D:\Mesobank_TMA\Mesobank_labels.csv')
        dets_path = Path(r'D:\Mesobank_TMA\mesobank_proj\detections')
        cents = pd.read_csv(r'D:\Mesobank_TMA\core_cents.csv')
        um_per_pix = 0.5034
        core_width = 1462
    #dets_list = list(dets_path.glob('*.geojson'))
    dets_list = list(dets_path.glob('*\*.geojson'))
    
    #1462 for mesobank, 2854 for meso
    um_per_pix=0.4415
    core_width = 2854
    SQ = SQLiteStore()
    #for dets in dets_list:    
        #SQ.add_from(dets)

    #SQ.commit()
    #SQ.dump(str(dets_path/'detections.db'))
    #core_cents=pd.read_csv(r'D:\QuPath_Projects\Meso_TMA\core_cents.csv')
    labels_df = pd.read_csv(labels_path)
    labels_df.set_index('Core', inplace=True)

    for core in dets_list:
        if core.stem not in labels_df.index:
            continue
        label = {'epithelioid': 'E', 'biphasic': 'B', 'sarcomatoid': 'S', 'desmoplastic': 'D'}[labels_df.loc[core.stem]['labels'].lower()]
        #label = labels_df.loc[core.stem]['labels']    #for meso
        #anns = SQ.query(where = f'props["Parent"] == "{core}"')
        SQ = SQLiteStore()
        top_left = cents.loc[core.stem.split('_')[0]][['Centroid X µm','Centroid Y µm']].values / um_per_pix - core_width/2
        with open(core, 'r') as f:
            anns = json.load(f)
            for ann in anns['features']:
                props = {pair['name']: pair['value'] if not math.isnan(pair['value']) else 0 for pair in ann['properties']['measurements']}
                poly = Polygon(ann['nucleusGeometry']['coordinates'][0])
                #keep these in slide space to extract resnet feats from etc
                props['Centroid X µm'] = poly.centroid.x
                props['Centroid Y µm'] = poly.centroid.y
                #make poly contour pts relative to core image top left for vis
                poly = translate(poly, -top_left[0], -top_left[1])
                SQ.append(Annotation(poly, props))
        
        SQ.commit()
        SQ.dump(str(dets_path/'stores'/f'{core.stem}_{label}.db'))


if __name__=="__main__":
    dataset = 'mesobank'
    if dataset == 'mesobank':
        dets_path = Path(r'D:\Mesobank_TMA\mesobank_proj\detections\stores')
        cents = pd.read_csv(r'D:\Mesobank_TMA\core_cents.csv')
        um_per_pix = 0.5034
        core_width = 1462
    else:
        dets_path = Path(r'D:\QuPath_Projects\Meso_TMA\detections\stores')
        cents = pd.read_csv(r'D:\QuPath_Projects\Meso_TMA\core_cents.csv')
        um_per_pix=0.4415
        core_width = 2854
    #dets_list = list(dets_path.glob('*.geojson'))
    dets_list = list(dets_path.glob('*.db'))
    cents.set_index('Name', inplace=True)
    for core in dets_list:
        if core.stem.split('_')[0] not in cents.index:
            continue
        print(f'processing core {core.stem}')
        SQ = SQLiteStore(core) 
        top_left = cents.loc[core.stem.split('_')[0]][['Centroid X µm','Centroid Y µm']].values / um_per_pix - core_width/2
        to_core_space(SQ, top_left)


