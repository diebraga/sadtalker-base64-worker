'''
The MIT License (MIT)
Copyright 2024 Dominic Powers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import warnings
warnings.filterwarnings("ignore")

import base64
import runpod
from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
import urllib.request

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

from utils.file_utils import download_file, map_network_volume, str2bool

# Checkpoints that must be present before the worker accepts jobs.
# These are downloaded at startup if not already on disk (e.g. on a fresh cold start).
CHECKPOINT_FILES = [
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar',
     '/app/SadTalker/checkpoints/mapping_00109-model.pth.tar'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar',
     '/app/SadTalker/checkpoints/mapping_00229-model.pth.tar'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors',
     '/app/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors'),
    ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors',
     '/app/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth',
     '/app/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
     '/app/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth'),
    ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
     '/app/SadTalker/gfpgan/weights/GFPGANv1.4.pth'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
     '/app/SadTalker/gfpgan/weights/parsing_parsenet.pth'),
]


def ensure_checkpoints():
    """Download any missing checkpoint files. Skips files that already exist."""
    os.makedirs('/app/SadTalker/checkpoints', exist_ok=True)
    os.makedirs('/app/SadTalker/gfpgan/weights', exist_ok=True)
    for url, path in CHECKPOINT_FILES:
        if os.path.exists(path):
            print(f'[SadTalker][startup]: {os.path.basename(path)} already present, skipping.')
            continue
        print(f'[SadTalker][startup]: Downloading {os.path.basename(path)} from {url} ...')
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'[SadTalker][startup]: Saved {path} ({size_mb:.1f} MB)')
        except Exception as e:
            print(f'[SadTalker][ERROR]: Failed to download {url}: {e}')
            return False, str(e)
    print('[SadTalker][startup]: All checkpoints ready.')
    return True, None


def generate_video(args):
    try:
        old_version = False
        checkpoint_dir = '/app/SadTalker/checkpoints'
        pic_path = args['source_image']
        audio_path = args['driven_audio']
        save_dir = os.path.join(args['result_dir'], strftime('%Y_%m_%d_%H.%M.%S'))
        os.makedirs(save_dir, exist_ok=True)

        pose_style = args.get('pose_style', os.getenv('DEFAULT_POSE_STYLE', 45))
        device = args.get('device', os.getenv('DEFAULT_DEVICE', 'cuda'))
        batch_size = args.get('batch_size', int(os.getenv('DEFAULT_BATCH_SIZE', 2)))
        input_yaw_list = args.get('input_yaw', os.getenv('DEFAULT_INPUT_YAW', None))
        input_pitch_list = args.get('input_pitch', os.getenv('DEFAULT_INPUT_PITCH', None))
        input_roll_list = args.get('input_roll', os.getenv('DEFAULT_INPUT_ROLL', None))
        ref_eyeblink = args.get('ref_eyeblink', os.getenv('DEFAULT_REF_EYEBLINK_URL', None))
        ref_pose = args.get('ref_pose', os.getenv('DEFAULT_REF_POSE_URL', None))
        size = args.get('size', int(os.getenv('DEFAULT_SIZE', 512)))
        preprocess = args.get('preprocess', os.getenv('DEFAULT_PREPROCESS', 'full'))
        still = args.get('still', str2bool(os.getenv('DEFAULT_STILL', 'True')) if 'DEFAULT_STILL' in os.environ else True)
        face3d = args.get('face3dvis', str2bool(os.getenv('DEFAULT_FACE3DVIS', 'False')) if 'FACE3DVIS' in os.environ else False)
        expression_scale = args.get('expression_scale', float(os.getenv('DEFAULT_EXPRESSION_SCALE', 1.0)))
        enhancer = args.get('enhancer', os.getenv('DEFAULT_ENHANCER', 'gfpgan'))
        background_enhancer = args.get('background_enhancer', os.getenv('DEFAULT_BACKGROUND_ENHANCER', None))

        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

        preprocess_model = CropAndExtract(sadtalker_paths, device)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('[SadTalker][source]: 3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            pic_path, first_frame_dir, preprocess, source_image_flag=True, pic_size=size
        )

        if first_coeff_path is None:
            return None, "[SadTalker][Error]: Can't get the coeffs of the input"

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('[SadTalker][blink]: 3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
                ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False
            )
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('[SadTalker][pose]: 3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ = preprocess_model.generate(
                    ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False
                )
        else:
            ref_pose_coeff_path = None

        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        if face3d:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, input_yaw_list,
            input_pitch_list, input_roll_list, expression_scale=expression_scale,
            still_mode=still, preprocess=preprocess, size=size
        )

        result = animate_from_coeff.generate(
            data, save_dir, pic_path, crop_info, enhancer=enhancer,
            background_enhancer=background_enhancer, preprocess=preprocess, img_size=size
        )

        output_video_path = shutil.move(result, save_dir + '.mp4')

        with open(output_video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')

        shutil.rmtree(save_dir)
        os.remove(output_video_path)

        return video_base64, None

    except Exception as e:
        return None, e


def handler(job):
    job_input = job['input']
    job_input['result_dir'] = 'results'

    input_image_url = job_input.get('input_image_url')
    input_audio_url = job_input.get('input_audio_url')
    ref_eyeblink_url = job_input.get('ref_eyeblink_url')
    ref_pose_url = job_input.get('ref_pose_url')

    if not input_image_url:
        return {'error': '"input_image_url" is required in job input.'}

    if not input_audio_url:
        return {'error': '"input_audio_url" is required in job input.'}

    job_input['source_image'], error = download_file(input_image_url, 'input_image.png')
    if error:
        return {'error': f'Could not download input_image_url: {error}'}

    job_input['driven_audio'], error = download_file(input_audio_url, 'input_audio.wav')
    if error:
        return {'error': f'Could not download input_audio_url: {error}'}

    if ref_eyeblink_url:
        job_input['ref_eyeblink'], error = download_file(ref_eyeblink_url, 'eyeroll.mp4')
        if error:
            print(f'[SadTalker][WARNING]: Could not download ref_eyeblink: {error}')

    if ref_pose_url:
        job_input['ref_pose'], error = download_file(ref_pose_url, 'ref_pose.mp4')
        if error:
            print(f'[SadTalker][WARNING]: Could not download ref_pose: {error}')

    if torch.cuda.is_available() and job_input.get('device') != ['cpu']:
        job_input['device'] = 'cuda'
    else:
        job_input['device'] = 'cpu'

    print(f'[SadTalker][device]: {job_input["device"]}')

    result, error = generate_video(job_input)

    if error:
        return {'error': f'generate_video failed: {error}'}
    else:
        return {'output_video_base64': result}


if __name__ == "__main__":
    result, error = map_network_volume()
    if error:
        print(f'[SadTalker][WARNING]: Could not map network volume: {error}')

    # Download checkpoints at startup if not already on disk.
    # (Checkpoints are no longer baked into the Docker image to keep it small.)
    ok, error = ensure_checkpoints()
    if not ok:
        print(f'[SadTalker][ERROR]: Failed to download checkpoints: {error}')
        sys.exit(1)

    runpod.serverless.start({'handler': handler})
