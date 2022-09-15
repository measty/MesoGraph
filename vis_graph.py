
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import layout
from bokeh.models import Slider, Toggle, Dropdown, Div, ColumnDataSource
from bokeh.models.callbacks import CustomJS
#from wasabi import change_pixel
from bokeh.models.mappers import EqHistColorMapper
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlGn11
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models.tools import TapTool
from utils import toNumpy, toTensor
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from gnn_explain import explain_net, sort_feats

"""vis stuff. Not really needed now contours are in annotation store.
But keep for reference."""



def change_pixel(mask,i,j,val):
    if val==255:
        val=0.01
    
    if mask[i,j]==255:
        mask[i,j]=val

        for a,b in zip([0,0,1,-1],[-1,1,0,0]):
            #if mask[i+a,j+b,0]==255:
                #mask[i+a,j+b,:]=val
            if not (i+a<0 or i+a>=mask.shape[0] or j+b<0 or j+b>=mask.shape[1]):
                change_pixel(mask,i+a,j+b, val)
    else:
        return


def bokeh_plot(g):

    Xn = toNumpy(g.coords)     # get coordinates
    Wn= np.array([e for e in toNumpy(g.edge_index.t())])
    xdf=pd.DataFrame(toNumpy(g.x),columns=g.feat_names[0])
    c=g.c
    core=g.core[0]
    feat_names=pd.Index(g.feat_names[0])
    print(f'creating vis for core: {core}...')
    tx=f'core {core}, true label is: {g.type_label}, score is: {g.z[1].item()/(g.z[0].item()+0.00001)}'

    TOOLTIPS=[
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("Top Feats", "@top_feats"),
      ]
    p = figure(title=tx,x_range=(0,2854), y_range=(0,2854), width=900, height=900, tooltips=TOOLTIPS)
    p.add_tools(TapTool())

    #mask=p.image_url(url=[f'D:\All_masks_inv\{core}.png'],x=0, y=2854, w=2854, h=2854, anchor="top_left", global_alpha=0.35)
    
    with Image.open(Path(f'D:\All_cores\{core}.jpg')) as imfile:
        blah=np.array(imfile.convert("RGBA"))
        blah_rgb=blah.view(dtype=np.uint32).reshape(blah.shape[:-1])

    blah_rgb=np.flipud(blah_rgb)
    #coreim=p.image_url(url=[f'D:\All_cores\{core}.jpg'],x=0, y=2854, w=2854, h=2854, anchor="top_left")
    coreim=p.image_rgba(image=[blah_rgb],x=0, y=0, dw=2854, dh=2854)

    fmask=Image.open(Path(f'D:\All_masks_tiff\{core}.tiff'))
    im=np.array(fmask, dtype=np.float32)
    im[im==1]=256
    im[im==2]=0
    im=im-1
    #im=255-im
    #im_rgba=np.concatenate((np.repeat(im[:,:,None],3,axis=2), 255*np.ones((im.shape[0], im.shape[1],1), dtype=np.uint8)), axis=2)
    #im_rgb=np.repeat(im[:,:,None],3,axis=2)
    #I = Image.fromarray(im_rgb).convert("RGBA")
    pal=[[1-i/20,i/20,0] for i in range(20)]
    cvals=[]
    for i in range(len(Xn)):
        cvals.append(0.5+(c[i,0]**2-c[i,1]**2)/2)
        if c[i,0]<0.4 and c[i,1]<0.4:
            cval=101
        else:
            cval=cvals[-1]
        change_pixel(im, int(Xn[i,1]), int(Xn[i,0]), cval)

    xdf['score']=cvals
    xdf['ep_score']=c[:,1]
    xdf['s_score']=c[:,0]
    xdf['x']=Xn[:,0]
    xdf['y']=2854-Xn[:,1]
    sorted_feats=sort_feats(xdf, feat_names, mutual_info_regression)
    sorted_feats.extend(['ep_score','s_score'])

    #gnn explainer stuff
    topN, topN_score, node_imp = explain_net(g)
    xdf['imp']=node_imp
    top_feats,top_scores=[],[]
    for k in range(topN.shape[0]):
        #top_str='\r\n'.join(feat_names[topN[k,:]])
        #top_feats.append(top_str)
        top_feats.append(feat_names[topN[k,:]])
        top_scores.append(topN_score[k,:])


    xdf['top_feats']=top_feats
    xdf['top_scores']=top_scores
    sorted_feats=['imp','top_feats']+sorted_feats
    
    im=np.flipud(im)
    fmask.close()
    #cmapper=LinearColorMapper(low=0,high=np.max(im[im<255]), palette='RdYlGn11', low_color=(255,255,255,0))
    
    highv=np.max(im[im<100])
    lowv=np.min(im[im>=0])
    mdiff=np.maximum(highv-0.5,0.5-lowv)
    mdiff=np.maximum(mdiff,0.4)
    lowv,highv=0.5-mdiff,0.5+mdiff
    fds=ColumnDataSource(xdf)

    cmapper=LinearColorMapper(low=lowv,high=highv, palette='RdYlGn11', high_color=(255,255,255,1) ,low_color=(255,255,255,0))
    drop_ehist_mapper=EqHistColorMapper(low=lowv,high=highv, palette='RdYlGn11', high_color=(255,255,255,1) ,low_color=(255,255,255,0))
    drop_lin_mapper=LinearColorMapper(low=lowv,high=highv, palette='RdYlGn11', high_color=(255,255,255,1) ,low_color=(255,255,255,0))
    circ_cmapper=linear_cmap(field_name='score', palette='RdYlGn11' ,low=0 ,high=1, high_color=(255,255,255,1) ,low_color=(255,255,255,0))
    #mask=p.image(image=[im],x=0, y=0, dw=2854, dh=2854, global_alpha=0.35, color_mapper=EqHistColorMapper('RdYlGn3'))
    mask=p.image(image=[im],x=0, y=0, dw=2854, dh=2854, global_alpha=0.45, color_mapper=cmapper)
    #mask=p.image_rgba(image=[np.array(I)],x=0, y=0, dw=2854, dh=2854, global_alpha=0.35)
    edges=p.segment(x0=Xn[Wn[:,0],0], y0=2854-Xn[Wn[:,0],1],
                x1=Xn[Wn[:,1],0], y1=2854-Xn[Wn[:,1],1],
                line_width=0.6)
    nodes=p.circle(x='x',y='y',color=circ_cmapper,source=fds, radius=3.5)

    menu = list(zip(sorted_feats,sorted_feats))
    drop_i = Dropdown(label="Select Feat (inf)", button_type="warning", menu=menu)
    sorted_feats=sort_feats(xdf, feat_names, f_regression)
    menu = list(zip(sorted_feats,sorted_feats))
    drop_f = Dropdown(label="Select Feat (f1)", button_type="warning", menu=menu)

    s2 = ColumnDataSource(data=dict(names=[1,2,3,4,5,6,7,8,9,10], scores=[0.1,0.2,0.3,0.1,0.1,0.24,0.5,0.4,0.3,0.2]))
    p2 = figure(width=700, height=500, x_range=(0, 1), y_range=(0,11))
    bar = p2.hbar(y='names', height=0.5, left=0, right='scores', color="navy", source=s2)
    #p2.yaxis.ticker = [1,2,3,4,5,6,7,8,9,10]

    topN_code = '''if (cb_data.source.selected.indices.length > 0){
            lines.visible = true;
            var selected_index = cb_data.source.selected.indices[0];
            lines.data_source.data['y'] = lines_y[selected_index]
            lines.data_source.change.emit(); 
          }'''

    fds.selected.js_on_change('indices', CustomJS(args=dict(s1=fds, s2=s2, ax=p2.yaxis[0], b=bar), code=r"""
        const inds = cb_obj.indices;
        const d1 = s1.data;
        const d2 = s2.data;
        console.log(d2)
        console.log(inds)
        const feats = d1['top_feats'][inds[0]]
        //console.log(p.y_range)
        //p.y_range.factors=feats
        const od={1: 'a', 2: 'a',3: 'a',4: 'a',5: 'a',6: 'a',7: 'a',8: 'a',9: 'a',10: 'a'}
        const scores = d1['top_scores'][inds[0]]
        //d2['names'] = []
        d2['scores'] = scores
        //console.log(p)
        console.log(feats)
        console.log(scores)
        for (let i = 0; i < feats.length; i++) {
            //d2['names'].push(feats[i])
            //d2['scores'].push(scores[i])
            od[i]=feats[i]
        }
        ax.major_label_overrides = od
        console.log(d2)
        s2.change.emit();
    """)
)

    #p.select(TapTool).callback=

    slider = Slider(
        title="Adjust alpha",
        start=0,
        end=1,
        step=0.05,
        value=0.5
    )
    toggle1 = Toggle(label="Show Mask", button_type="success")
    toggle2 = Toggle(label="Show Lines", button_type="success")
    toggle3 = Toggle(label="Show Dots", button_type="success")
    div = Div(text="""dots: score""",
        width=300, height=60)
    #drop=Dropdown(label='choose core', button_type='warning', menu=[('3-B','3-B'),('3-C','3-C'),('3-D','3-D')])

    callback = CustomJS(args=dict(m=mask, s=slider), code="""

    // JavaScript code goes here

    const a = 10;
    console.log('checkbox_button_group: active=' + this.active, this.toString())
    // the model that triggered the callback is cb_obj:
    const b = cb_obj.active;
    
    // models passed as args are automagically available
    if (this.active) {
        m.glyph.global_alpha = 0;
    } else {
        m.glyph.global_alpha = s.value;
    };
    console.log('b is: ' + b)

    """)
    callback2 = CustomJS(args=dict(e=edges, s=slider), code="""
        if (this.active) {
            e.visible = false;
        } else {
            e.glyph.line_alpha = s.value;
            e.visible = true;
        };
    """)
    callback3 = CustomJS(args=dict(n=nodes, s=slider), code="""
        if (this.active) {
            n.visible = false;
        } else {
            n.glyph.line_alpha = s.value;
            n.visible = true;
        };
    """)
    slidercb=CustomJS(args=dict(e=edges, m=mask, n=nodes, s=slider, mt=toggle1, et=toggle2), code="""
        if (et.active) {
            e.glyph.line_alpha = 0;
        } else {
            e.glyph.line_alpha = s.value;
        };
        if (mt.active) {
            m.glyph.global_alpha = 0;
        } else {
            m.glyph.global_alpha = s.value;
        };
        n.glyph.line_alpha = s.value;
        n.glyph.fill_alpha = s.value;

    """)

    #dropcb=CustomJS(args=dict(c=coreim), code="""
    #    c.glyph.url=`D:\All_cores\${this.item}.jpg`
    #
    #""")

    dropcb=CustomJS(args=dict(c=circ_cmapper, ehm=drop_ehist_mapper, lm=drop_lin_mapper, n=nodes,ds=fds, pal=RdYlGn11, p=div), code=r"""
        var low = Math.min.apply(Math,ds.data[this.item]);
        var high = Math.max.apply(Math,ds.data[this.item]);
        this.label=this.item
        p.text='dots:'+this.item
        console.log(this.item)
        c.field_name=this.item
        //var color_mapper = new Bokeh.LinearColorMapper({palette:pal, low:0, high:1});
        c.transform.update_data()
        if (this.item == 'ep_score' || this.item=='s_score') {
            var cm=lm
        }
        else {
            var cm=ehm
        }
        cm.low=low
        cm.high=high
        n.glyph.color = {field: this.item, transform: cm};
        n.glyph.fill_color = {field: this.item, transform: cm};
        n.glyph.line_color = {field: this.item, transform: cm};
        ds.change.emit();
    """)
    #<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/dev/bokeh-api-3.0.0dev1.min.js"></script>

    #slider.js_link("value", nodes.glyph , "fill_alpha")
    slider.js_link("value", nodes.glyph , "line_alpha")
    #slider.js_link("value", edges.glyph , "line_alpha")
    slider.js_on_change('value', slidercb)

    toggle1.js_on_click(callback)
    toggle2.js_on_click(callback2)
    toggle3.js_on_click(callback3)
    drop_i.js_on_event('menu_item_click',dropcb)
    drop_f.js_on_event('menu_item_click',dropcb)

    # create layout
    gr = layout(
        [
            [p, [slider, toggle1, toggle2, toggle3, drop_i, drop_f, div,p2]],
        ]
    )

    # show result
    output_file(filename=f"D:/Meso/Bokeh_core_temp/{core}_{g.type_label[0]}.html", title="TMA cores graph NN visualisation")
    save(gr)
    #show(gr)
    #html = file_html(gr, CDN, "my plot")
    #with open('D:/blah.html', 'w') as f:
        #f.write(html)
    print('save?')
