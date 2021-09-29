import cv2
import matplotlib

matplotlib.use('Agg')
import sys
import yaml
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
from cats_keypoints_detector.predict import predict_kps

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if not cpu:
        kp_detector.cuda()

    if not cpu:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_images, generator, kp_detector, cpu, relative=True, adapt_movement_scale=True):
    vid = cv2.VideoCapture(0)
    source_images_dict = dict()
    source_kps_dict = dict()
    current_show = 1

    with torch.no_grad():
        kp_driving_initial = None
        while True:
            ret, driving_frame = vid.read()
            driving_frame = resize(driving_frame, (256, 256))
            driving_frame = torch.tensor(np.array(driving_frame).astype(np.float32)).permute(2, 0, 1).unsqueeze(
                0)  # have to be 1, 3, 256, 256
            if not cpu:
                driving_frame = driving_frame.cuda()
            if kp_driving_initial is None:
                kp_driving_initial = kp_driving = kp_detector(driving_frame)  # todo: check this
            else:
                kp_driving = kp_detector(driving_frame)
            driv_img = img_as_ubyte(driving_frame[0].permute(1, 2, 0).cpu().numpy())
            result_imgs = [driv_img]
            for i in range(current_show):
                source = source_images_dict.get(i)
                if source is None:
                    source_image = imageio.imread(source_images[i])
                    source_image = resize(source_image, (256, 256))[..., :3]
                    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    if not cpu:
                        source = source.cuda()
                    source_images_dict[i] = source

                kp_source = source_kps_dict.get(i)
                if not kp_source:
                    image = np.copy(source[0].permute(1, 2, 0).cpu().numpy())  # todo: optimize it
                    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]

                    kp_array = predict_kps(source_image, cpu)

                    kp_array = ((kp_array * 2) / spatial_size) - 1
                    kp_source = kp_detector(source)
                    kp_source['value'][0] = kp_array
                    source_kps_dict[i] = kp_source

                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)

                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                img = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                img = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2RGB)
                result_imgs.append(img)
            cv2.imshow('frame', np.concatenate(result_imgs, 1))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(33) == ord('a'):
                print('Next!')
                current_show += 1
                if current_show > len(source_images):
                    break
        # Destroy all the windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator, kp_detector = load_checkpoints(opt.config, opt.checkpoint, opt.cpu)

    source_images = ['./animal_kps/photos/1.jpg', './animal_kps/photos/4.jpg', './animal_kps/photos/6.jpg', './animal_kps/photos/9.jpg']

    make_animation(source_images, generator, kp_detector, opt.cpu, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
