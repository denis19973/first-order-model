import json
import os
import time

import cv2
import matplotlib

matplotlib.use('Agg')
import yaml
import pyfakewebcam

from argparse import ArgumentParser

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

IMG_W = 256
IMG_H = 256
TIMEOUT_SEC = 2


def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_images, generator, kp_detector, relative=True, adapt_movement_scale=True, kp_arrays=None):
    vid = cv2.VideoCapture(0)
    fake_cam = pyfakewebcam.FakeWebcam('/dev/video2', IMG_W, IMG_H)

    source_images_dict = dict()
    timeout_sec = TIMEOUT_SEC
    source_kps_dict = dict()

    for current_show in range(11):
        source_image = imageio.imread(source_images[current_show])
        source_image = resize(source_image, (256, 256))[..., :3]
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.cuda()
        source_images_dict[current_show] = source

        image = np.copy(source[0].permute(1, 2, 0).cpu().numpy())  # todo: optimize it
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]

        kp_array = kp_arrays[current_show]

        value = ((kp_array * 2) / spatial_size) - 1
        kp_source = kp_detector(source)
        kp_source['value'][0] = torch.from_numpy(value).cuda()
        source_kps_dict[current_show] = kp_source

    current_show = 0
    first_cat = True
    dance_time = False
    start = time.time()
    with torch.no_grad():
        kp_driving_initial = None
        while True:
            if current_show == 8:
                current_show += 1
                continue
            end = time.time()
            elapsed = end - start
            if first_cat and elapsed > 45:  # first sumup
                first_cat = False
                current_show += 1
                # kp_driving_initial = None
                continue
            elif not first_cat and elapsed > timeout_sec:
                start = time.time()
                current_show += 1
                # kp_driving_initial = None
                if current_show > 10:
                    current_show = 0
                    timeout_sec = 2

            ret, driving_frame = vid.read()
            driving_frame = resize(driving_frame, (256, 256))
            cv2.imshow('frame', driving_frame)
            if cv2.waitKey(1) == 27:
                break

            driving_frame = torch.tensor(np.array(driving_frame).astype(np.float32)).permute(2, 0, 1).unsqueeze(
                0).cuda()  # have to be 1, 3, 256, 256
            if kp_driving_initial is None:
                kp_driving_initial = kp_driving = kp_detector(driving_frame)
            else:
                kp_driving = kp_detector(driving_frame)

            source = source_images_dict.get(current_show)
            if source is None:
                source_image = imageio.imread(source_images[current_show])
                source_image = resize(source_image, (256, 256))[..., :3]
                source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                source = source.cuda()
                source_images_dict[current_show] = source

            kp_source = source_kps_dict.get(current_show)
            if not kp_source:
                image = np.copy(source[0].permute(1, 2, 0).cpu().numpy())  # todo: optimize it
                spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]

                kp_array = kp_arrays[current_show]

                value = ((kp_array * 2) / spatial_size) - 1
                kp_source = kp_detector(source)
                kp_source['value'][0] = torch.from_numpy(value).cuda()
                source_kps_dict[current_show] = kp_source

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            img = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            # img = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2RGB)
            img = img_as_ubyte(img)
            # cv2.imshow('frame', np.concatenate(result_imgs, 1))
            fake_cam.schedule_frame(img)
            # Press Q on keyboard to  exit
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)

    source_images = list(os.listdir('./animal_kps/photos'))
    source_images.sort(key=lambda f: int(f.split('.')[0]))
    source_images = [os.path.join('./animal_kps/photos', s) for s in source_images]
    with open('./animal_kps/keypoints.json') as f:
        kps_dict = json.load(f)

    kp_arrays = []
    for img_name in source_images:
        kp_arrays.append(kps_dict[os.path.basename(img_name)])
    kp_arrays = np.array(kp_arrays)

    make_animation(source_images, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
                   kp_arrays=kp_arrays)
