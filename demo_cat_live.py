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

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


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
    source_images_dict = dict()
    source_kps_dict = dict()
    current_show = 1

    with torch.no_grad():
        kp_driving_initial = None
        while True:
            ret, driving_frame = vid.read()
            driving_frame = resize(driving_frame, (256, 256))
            driving_frame = torch.tensor(np.array(driving_frame).astype(np.float32)).permute(2, 0, 1).unsqueeze(
                0).cuda()  # have to be 1, 3, 256, 256
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
                    source = source.cuda()
                    source_images_dict[i] = source

                kp_source = source_kps_dict.get(i)
                if not kp_source:
                    image = np.copy(source[0].permute(1, 2, 0).cpu().numpy())  # todo: optimize it
                    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]

                    kp_array = kp_arrays[i]

                    value = ((kp_array * 2) / spatial_size) - 1
                    kp_source = kp_detector(source)
                    kp_source['value'][0] = torch.from_numpy(value).cuda()
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

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)

    kp_array_cat_4 = [  # cat 4
        [169.94967651, 125.91460419],
        [132.97007751, 170.11863708],  #
        [164.81164551, 75.29729462],  #
        [76.58087921, 134.62646484],  #
        [137.82058716, 113.89077759],  #
        [171.97190857, 156.13345337],  #
        [159.37608337, 111.58245087],  #
        [119.78703308, 178.02449036],  #
        [91.05801392, 81.95692444],
        [125.39608765, 68.21273041],  #
    ]

    kp_array_cat_3 = [  # cat 3
        [173.80639648, 102.18058777],  #
        [131.84580994, 135.38687134],  #
        [156.41783142, 45.39122009],  #
        [75.52799988, 102.896698],
        [137.84056091, 79.5743866],  #
        [183.74308777, 132.88595581],  #
        [160.40460205, 82.05131531],  #
        [116.81054688, 149.38078308],  #
        [85.8482666, 61.16685486],  #
        [127.7477417, 46.89950562],  #
    ]

    kp_array_dog = [  # dog 0
        [181.91725159, 144.80281067],  #
        [125.9697113, 165.90283203],  #
        [165.03390503, 92.53762817],  #
        [77.48940277, 154.43751526],  #
        [138.09819031, 117.78536224],  #
        [178.91503906, 167.09927368],  #
        [168.54492188, 125.39859772],  #
        [107.46125031, 178.29876709],  #
        [71.40492249, 105.26499939],  #
        [117.70788574, 91.53234863],  #
    ]
    source_images = ['./cat_4.jpg', './cat_3.jpg', './dog_0.jpg']
    kp_arrays = np.array([kp_array_cat_4, kp_array_cat_3, kp_array_dog])

    make_animation(source_images, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
                   kp_arrays=kp_arrays)
