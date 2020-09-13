
import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="./e2e_faster_rcnn_R_50_FPN_1x.pth",
    help="path to detectron pretrained weight(.pth)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/faster_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_faster_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = torch.load(DETECTRON_PATH)
newdict = _d
print(_d["model"].keys())
newdict['model'] = removekey(_d['model'],
                             ['module.roi_heads.box.predictor.cls_score.bias', 'module.roi_heads.box.predictor.cls_score.weight', 'module.roi_heads.box.predictor.bbox_pred.bias', 'module.roi_heads.box.predictor.bbox_pred.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))

# If the scheduler are also need to be removed, run the code below: 
# ori = torch.load("./pretrained_model/faster_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth")

# new = {"model": ori["model"]}
# torch.save(new, "./pretrained_model/faster_rcnn_R-50-FPN_1x_detectron_no_scheduler.pth")