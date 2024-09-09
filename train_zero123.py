import json
import os
import tqdm

json_path = '/mnt/data/datasets/objaverse/uid_to_objects.json'
with open(json_path, 'r') as f:
    objects_list = list(json.load(f).items())
for i, (uid, glb_path) in tqdm.tqdm(enumerate(objects_list)):
    shell_cmd = f'python train.py -s /mnt/data/datasets/zero123/views_release_10000/{uid} --iterations 30000 --freeze_xyz --densify_until_iter -1 --no_tqdm --eval --model_path output_zero123/{uid} --obj_path {glb_path} --white_background'
    os.system(shell_cmd)