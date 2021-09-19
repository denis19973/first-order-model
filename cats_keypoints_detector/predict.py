import torch
import torch.utils.data
from cats_keypoints_detector.detector import Detector
from cats_keypoints_detector.mapper import Mapper
from cats_keypoints_detector.utils import IMGNET_TRANSFORM
from skimage.transform import resize
import numpy as np


DETECTOR_STATE_PATH = 'checkpoints/detector_state.pth'
MAPPER_STATE_PATH = 'checkpoints/mapper_state.pth'
DETECTOR_SIZE = (224, 224)
MAPPER_INPUT_SIZE = 512


def predict_kps(img, cpu, size=256):
    img = resize(img, DETECTOR_SIZE)[..., :3]
    img = img.astype(np.float32)

    img = IMGNET_TRANSFORM(img)
    img = img.unsqueeze(0)

    model_d = Detector()
    if cpu:
        model_d.load_my_state_dict(torch.load(DETECTOR_STATE_PATH, map_location=torch.device('cpu')))
    else:
        model_d.load_my_state_dict(torch.load(DETECTOR_STATE_PATH))
    model_d.eval()

    kbs = model_d(img)
    kbs = (kbs * (size // 2)) + (size // 2)
    kbs = kbs.reshape(-1, 18)

    model_m = Mapper(18, 20)
    if cpu:
        model_m.load_state_dict(torch.load(MAPPER_STATE_PATH))
    else:
        model_m.load_state_dict(torch.load(MAPPER_STATE_PATH, map_location=torch.device('cuda')))
    model_m.eval()

    kbs = model_m(kbs).squeeze()
    
    # kbs = kbs / MAPPER_INPUT_SIZE * size
    kbs[[0, 8]] = kbs[[8, 0]]
    kbs[[6, 2]] = kbs[[2, 6]]
    kbs[[7, 1]] = kbs[[1, 7]]
    kbs[[3, 5]] = kbs[[5, 3]]

    return kbs
