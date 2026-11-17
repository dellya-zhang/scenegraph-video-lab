"""Optional pretrained inference adapter. Not used in the CPU demo; not GPU validated."""
import argparse
import json
from pathlib import Path


def graph_to_prompt(graph):
    names = {n['id']: n['label'].replace('_', ' ') for f in graph['frames'] for n in f['nodes']}
    lines = ['A fixed camera records colored geometric objects on a light grid background.']
    for label, frame in [('Initially', graph['frames'][0]), ('Finally', graph['frames'][-1])]:
        relations = [f"{names[e['source']]} is {e['relation'].replace('_',' ')} {names[e['target']]}"
                     for e in frame['edges'] if e['relation'] in ('left_of', 'right_of', 'above')]
        lines.append(label + ': ' + '; '.join(relations) + '.')
    return ' '.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', default='docs/assets/graph.json')
    p.add_argument('--image', default='docs/assets/condition.png')
    p.add_argument('--output', default='outputs/diffusion.mp4')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    prompt = graph_to_prompt(json.loads(Path(a.graph).read_text(encoding='utf-8')))
    print(prompt)
    if a.dry_run:
        return
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import load_image, export_to_video
    if not torch.cuda.is_available():
        raise RuntimeError('This optional adapter requires CUDA. Use --dry-run on CPU.')
    pipe = CogVideoXImageToVideoPipeline.from_pretrained('THUDM/CogVideoX-5b-I2V', torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    image = load_image(a.image).resize((720, 480))
    result = pipe(image=image, prompt=prompt, num_frames=49, num_inference_steps=50,
                  guidance_scale=6, generator=torch.Generator('cpu').manual_seed(a.seed)).frames[0]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result, a.output, fps=8)


if __name__ == '__main__':
    main()
