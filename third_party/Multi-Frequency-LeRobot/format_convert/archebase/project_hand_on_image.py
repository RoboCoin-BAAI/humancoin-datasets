"""Project left-hand 3D landmarks onto a matching left-camera video frame."""
import argparse
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from format_convert.field_naming import HAND_LANDMARK_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--frame',type=int,default=0)
    args=parser.parse_args()
    root=args.dataset
    data=root/'data/chunk-000/episode_000000'
    camera=pq.read_table(data/'observation.images.head_left.parquet')['timestamp'].to_numpy()
    if not 0 <= args.frame < len(camera):
        parser.error('Frame index outside video timeline')
    t=float(camera[args.frame])
    table=pq.read_table(data/'observation.hand_left_points.parquet')
    j=int(np.argmin(abs(table['timestamp'].to_numpy()-t)))
    xyz=np.array([[table[f'{name}_{axis}'][j].as_py() for axis in ('x','y','z')]
                  for name in HAND_LANDMARK_NAMES],dtype=np.float64)
    origin=json.loads((root/'meta/archebase_source.json').read_text(encoding='utf-8'))['timestamp_origin_ns']
    cals=json.loads((root/'calibrations/camera_info.json').read_text(encoding='utf-8'))['head_left']
    eligible=[c for c in cals if (c['timestamp_ns']-origin)/1e9<=t]
    cal=(eligible[-1] if eligible else cals[0])['camera_info']
    k=np.array(cal['k']).reshape(3,3)
    d=np.array(cal['d'])
    if cal['distortion_model']=='equidistant':
        uv,_=cv2.fisheye.projectPoints(xyz.reshape(-1,1,3),np.zeros(3),np.zeros(3),k,d)
    elif cal['distortion_model'] in ('plumb_bob','rational_polynomial'):
        uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),k,d)
    else:
        raise ValueError('Unsupported distortion model: '+cal['distortion_model'])
    uv=uv.reshape(-1,2)
    image=None
    with av.open(str(root/'videos/chunk-000/observation.images.head_left/episode_000000.mp4')) as c:
        for i,frame in enumerate(c.decode(video=0)):
            if i==args.frame:
                image=frame.to_ndarray(format='rgb24')
                break
    if image is None:
        raise ValueError('Video frame unavailable')
    source2d=pq.read_table(data/'observation.hand_left_points_2d.parquet')
    s=int(np.argmin(abs(source2d['timestamp'].to_numpy()-t)))
    source_uv=np.array([[source2d[f'{n}_{a}'][s].as_py() for a in ('u','v')] for n in HAND_LANDMARK_NAMES])
    distances=np.linalg.norm(uv-source_uv,axis=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(18,8),gridspec_kw=dict(width_ratios=[1.4,1]))
    colors=['#ff9f1c','#00bfff','#00ff88','#ff5555','#df88ff']
    for ax in axes:
        ax.imshow(image)
        for chain,color in zip(((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)),colors):
            v=uv[list(chain)]
            ax.plot(v[:,0],v[:,1],'-o',color=color,linewidth=2,markersize=4)
        ax.axis('off')
    for i,name in enumerate(HAND_LANDMARK_NAMES):
        axes[0].annotate(str(i),uv[i],xytext=(4,4),textcoords='offset points',color='white',fontsize=8,
                         bbox=dict(facecolor='black',alpha=.5,pad=.3))
        axes[1].annotate(f'{i}: {name}',uv[i],xytext=(6,4),textcoords='offset points',color='white',fontsize=9,
                         bbox=dict(facecolor='black',alpha=.6,pad=.5))
    lo,hi=uv.min(0)-65,uv.max(0)+95
    axes[1].set_xlim(lo[0],hi[0]);axes[1].set_ylim(hi[1],lo[1])
    axes[0].set_title('Left camera: projected 3D landmarks')
    axes[1].set_title('Detail: canonical landmark names')
    fig.suptitle(f'Frame {args.frame} | t={t:.3f}s | equidistant projection | source XYZ treated as left-camera coordinates\n'
                 f'Comparison with source 2D: mean {distances.mean():.3f}px, max {distances.max():.3f}px')
    fig.tight_layout()
    dest=root/'preview'
    dest.mkdir(exist_ok=True)
    output=dest/f'hand_left_projected_{args.frame:06d}.png'
    fig.savefig(output,dpi=150)
    report=dict(frame=args.frame,camera_time_s=t,hand_time_s=table['timestamp'][j].as_py(),
                coordinate_assumption='source XYZ interpreted in left-camera optical coordinates; no extrinsic applied',
                source_2d_mean_error_px=float(distances.mean()),source_2d_max_error_px=float(distances.max()),
                all_points_in_front=bool((xyz[:,2]>0).all()),projected_uv=uv.tolist())
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(output)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
