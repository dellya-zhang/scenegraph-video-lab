"""CPU reference: pixels -> tracked boxes -> temporal graph -> editable video.

The detector only recognizes the three fixture colors; it is NOT a learned model.
"""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

COLORS = {'red': (235, 76, 92), 'blue': (60, 130, 245), 'green': (34, 185, 140)}


def background(w=640, h=360):
    im = np.full((h, w, 3), (241, 245, 249), np.uint8)
    for x in range(0, w, 40):
        cv2.line(im, (x, 0), (x, h), (224, 232, 240), 1)
    for y in range(0, h, 40):
        cv2.line(im, (0, y), (w, y), (224, 232, 240), 1)
    return im


def fixture(n=60):
    frames = []
    for i in range(n):
        im = background()
        # Parallel trajectories reverse left_of/right_of without occlusion.
        x = 40 + int(480 * i / (n - 1))
        cv2.rectangle(im, (x, 70), (x + 48, 112), COLORS['red'][::-1], -1)
        cv2.rectangle(im, (560 - x, 170), (608 - x, 212), COLORS['blue'][::-1], -1)
        cv2.rectangle(im, (290, 270), (340, 310), COLORS['green'][::-1], -1)
        frames.append(im)
    return frames


def extract(frames, fps=12):
    if not frames or fps <= 0:
        raise ValueError('Need decoded frames and positive fps')
    h, w = frames[0].shape[:2]
    previous, next_id, timeline = {}, 1, []
    for fi, im in enumerate(frames):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.int16)
        nodes, current = [], {}
        for color, value in COLORS.items():
            mask = (np.linalg.norm(rgb - np.array(value), axis=2) < 60).astype(np.uint8)
            count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
            candidates = sorted(range(1, count), key=lambda k: stats[k, 4], reverse=True)
            used = set()
            for k in candidates:
                x, y, bw, bh, area = map(int, stats[k])
                if area < 80:
                    continue
                center = centers[k]
                matches = [(np.linalg.norm(center - p['center']), tid) for tid, p in previous.items()
                           if p['color'] == color and tid not in used]
                distance, tid = min(matches, default=(float('inf'), -1))
                if distance > 80:
                    tid, next_id = next_id, next_id + 1
                used.add(tid)
                current[tid] = {'center': center, 'color': color}
                nodes.append({'id': tid, 'label': color + '_object', 'color': color,
                              'bbox': [round(x/w, 5), round(y/h, 5), round(bw/w, 5), round(bh/h, 5)]})
        edges = []
        for a in nodes:
            for b in nodes:
                if a['id'] >= b['id']:
                    continue
                ax, ay, aw, ah = a['bbox']; bx, by, bw, bh = b['bbox']
                dx, dy = ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2
                if abs(dx) > .025:
                    edges.append({'source': a['id'], 'target': b['id'], 'relation': 'left_of' if dx < 0 else 'right_of'})
                if abs(dy) > .025:
                    edges.append({'source': a['id'], 'target': b['id'], 'relation': 'above' if dy < 0 else 'below'})
                if np.hypot(dx, dy) < .30:
                    edges.append({'source': a['id'], 'target': b['id'], 'relation': 'near'})
        timeline.append({'frame': fi, 'time': round(fi/fps, 4), 'nodes': nodes, 'edges': edges})
        previous = current
    return {'schema_version': '1.0', 'fps': fps, 'width': w, 'height': h,
            'provenance': 'color components + nearest-center tracking + geometric relations', 'frames': timeline}


def render(graph, drop_id=None):
    frames = []
    for frame in graph['frames']:
        im = background(graph['width'], graph['height'])
        for node in frame['nodes']:
            if node['id'] == drop_id:
                continue
            x, y, w, h = node['bbox']
            x, y, w, h = [round(v*s) for v, s in zip((x,y,w,h), (graph['width'],graph['height'])*2)]
            cv2.rectangle(im, (x,y), (x+w-1,y+h-1), COLORS[node['color']][::-1], -1)
        frames.append(im)
    return frames


def remove_object(graph, object_id):
    edited = copy.deepcopy(graph)
    for frame in edited['frames']:
        frame['nodes'] = [n for n in frame['nodes'] if n['id'] != object_id]
        frame['edges'] = [e for e in frame['edges'] if object_id not in (e['source'], e['target'])]
    return edited


def save_gif(frames, path, fps):
    images = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000/fps), loop=0)


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), []
    try:
        while len(frames) < 600:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
    finally:
        cap.release()
    if not frames:
        raise ValueError('Cannot decode input video')
    return frames, fps if fps > 0 else 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help='Optional video; only fixture colors are recognized')
    parser.add_argument('--out', type=Path, default=Path('docs/assets'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.input) if args.input else (fixture(), 12)
    graph = extract(frames, fps)
    edited = remove_object(graph, 1)
    for name, value in [('graph', graph), ('edited_graph', edited)]:
        (args.out / (name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
    for name, images in [('input', frames), ('reconstruction', render(graph)), ('edited', render(edited))]:
        save_gif(images, args.out / (name+'.gif'), fps)
    cv2.imwrite(str(args.out/'condition.png'), render(graph)[0])
    print(f'Exported {len(frames)} frames, {len(graph["frames"][0]["nodes"])} initial tracks -> {args.out}')


if __name__ == '__main__':
    main()
