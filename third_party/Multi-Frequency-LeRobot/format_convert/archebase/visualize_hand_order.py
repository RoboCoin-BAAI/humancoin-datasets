"""Preview a proposed 21-landmark mapping without modifying the dataset."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from format_convert.field_naming import HAND_LANDMARK_NAMES

FINGERS = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
COLORS = ['#ee7733','#0077bb','#009988','#cc3311','#aa4499']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    args = parser.parse_args()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    root = args.dataset
    data = root/'data/chunk-000/episode_000000'
    hands = {}
    for side in ('left','right'):
        t = pq.read_table(data/f'observation.hand_{side}_points.parquet')
        cols = [c for c in t.column_names if c not in ('timestamp','episode_index')]
        hands[side] = (t['timestamp'].to_numpy(),np.column_stack([t[c].to_numpy() for c in cols]).reshape(-1,21,3))
    times = hands['left'][0]
    selected = np.linspace(0,len(times)-1,21,dtype=int)
    fig = make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                        subplot_titles=['Left hand — proposed names','Right hand — proposed names'])

    def traces(index):
        result = []
        for side in ('left','right'):
            ts, points = hands[side]
            j = int(np.argmin(abs(ts-times[index])))
            p = points[j]-points[j,0]  # Translation only, source axes unchanged.
            for chain,color in zip(FINGERS,COLORS):
                v=p[list(chain)]
                result.append(go.Scatter3d(x=v[:,0],y=v[:,1],z=v[:,2],mode='lines',
                                          line=dict(color=color,width=6),showlegend=False,hoverinfo='skip'))
            result.append(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers+text',
                marker=dict(size=4,color='#222222'),text=[f'{i}: {n}' for i,n in enumerate(HAND_LANDMARK_NAMES)],
                textposition='top center',textfont=dict(size=11),showlegend=False,
                hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'))
        return result

    for i,trace in enumerate(traces(selected[0])):
        fig.add_trace(trace,row=1,col=1 if i<6 else 2)
    fig.frames=[go.Frame(name=str(i),data=traces(index),traces=list(range(12))) for i,index in enumerate(selected)]
    scene=dict(aspectmode='cube',xaxis=dict(title='source X (wrist-centered)',range=[-.25,.25]),
               yaxis=dict(title='source Y (wrist-centered)',range=[-.25,.25]),
               zaxis=dict(title='source Z (wrist-centered)',range=[-.25,.25]),uirevision='hand')
    fig.update_layout(title='Archebase 186460 — PROPOSED mapping, awaiting confirmation',
        scene=scene,scene2=scene,height=850,margin=dict(l=0,r=0,t=100,b=130),
        sliders=[dict(currentvalue=dict(prefix='Sample time: '),steps=[dict(label=f'{times[index]:.2f}s',
            method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))])
            for i,index in enumerate(selected)])],
        annotations=list(fig.layout.annotations)+[dict(x=.5,y=-.06,xref='paper',yref='paper',showarrow=False,
            text='Drag to rotate; scroll to zoom. Slider: 21 samples across episode. Connections and names are candidates.<br>'
                 'Both hands translated to their own wrist; source axes and point order unchanged. No dataset files modified.')])
    dest=root/'preview'
    dest.mkdir(exist_ok=True)
    fig.write_html(dest/'hand_landmark_order.html',include_plotlyjs=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    static=plt.figure(figsize=(16,8))
    for col,side in enumerate(('left','right'),1):
        ax=static.add_subplot(1,2,col,projection='3d')
        p=hands[side][1][0]-hands[side][1][0,0]
        for chain,color in zip(FINGERS,COLORS):
            v=p[list(chain)]
            ax.plot(v[:,0],v[:,1],v[:,2],'-o',color=color,markersize=4)
        for j,name in enumerate(HAND_LANDMARK_NAMES):
            ax.text(*p[j],f'{j}: {name}',fontsize=8)
        ax.set(xlabel='Source X',ylabel='Source Y',zlabel='Source Z',title=side+' hand (proposed labels)')
        ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),.01))
        ax.view_init(elev=-65,azim=-90)
    static.suptitle('Archebase 186460 | first frame | proposed mapping (unconfirmed)')
    static.tight_layout()
    static.savefig(dest/'hand_landmark_order.png',dpi=160)
    print(dest/'hand_landmark_order.html')
    print(dest/'hand_landmark_order.png')


if __name__=='__main__':
    main()
