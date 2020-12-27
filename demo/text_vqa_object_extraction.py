import os
import io
import json
import h5py
import numpy as np
import cv2
import torch
import detectron2
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image


def get_parser(self):
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", default="./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml", type=str, help="config path")
  parser.add_argument("--weight_path", default="../faster_rcnn_from_caffe_attr.pkl", type=str, help="pretrained model weight path")
  parser.add_argument("--img_index_file", type=str, help="Image index to id file")
  parser.add_argument("--img_folder", type=str, help="Image folder path")
  parser.add_argument("--tier", type=str, help="train, val, or test")
  parser.add_argument("--save_path", type=str, help="path to save object features")
  return parser


def load_model(config_path, weight_path):
  data_path = 'data/genome/1600-400-20'

  vg_classes = []
  with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
      for object in f.readlines():
          vg_classes.append(object.split(',')[0].lower().strip())
          
  vg_attrs = []
  with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
      for object in f.readlines():
          vg_attrs.append(object.split(',')[0].lower().strip())


  MetadataCatalog.get("vg").thing_classes = vg_classes
  MetadataCatalog.get("vg").attr_classes = vg_attrs

  cfg = get_cfg()
  cfg.merge_from_file(config_path)
  cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
  # VG Weight
  cfg.MODEL.WEIGHTS = weight_path
  predictor = DefaultPredictor(cfg)

  return predictor


def doit(raw_image, predictor):
  NUM_OBJECTS = 36
  with torch.no_grad():
    raw_height, raw_width = raw_image.shape[:2]
    # print("Original image size: ", (raw_height, raw_width))
    
    # Preprocessing
    image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
    # print("Transformed image size: ", image.shape[:2])
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": raw_height, "width": raw_width}]
    images = predictor.model.preprocess_image(inputs)
    
    # Run Backbone Res1-Res4
    features = predictor.model.backbone(images.tensor)
    
    # Generate proposals with RPN
    proposals, _ = predictor.model.proposal_generator(images, features, None)
    proposal = proposals[0]
    # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
    
    # Run RoI head for each proposal (RoI Pooling + Res5)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    features = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(
        features, proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
    # print('Pooled features size:', feature_pooled.shape)
    
    # Predict classes and boxes for each proposal.
    pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
    outputs = FastRCNNOutputs(
        predictor.model.roi_heads.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        predictor.model.roi_heads.smooth_l1_beta,
    )
    probs = outputs.predict_probs()[0]
    boxes = outputs.predict_boxes()[0]
    
    #add for test
    #print(outputs.predict_boxes())
    #print(outputs.predict_boxes()[0].shape)
    #add for test

    attr_prob = pred_attr_logits[..., :-1].softmax(-1)
    max_attr_prob, max_attr_label = attr_prob.max(-1)
    
    # Note: BUTD uses raw RoI predictions,
    #       we use the predicted boxes instead.
    # boxes = proposal_boxes[0].tensor    
    
    # NMS
    for nms_thresh in np.arange(0.5, 1.0, 0.1):
        instances, ids = fast_rcnn_inference_single_image(
            boxes, probs, image.shape[1:], 
            score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
        )
        if len(ids) == NUM_OBJECTS:
            break
            
    instances = detector_postprocess(instances, raw_height, raw_width)
    roi_features = feature_pooled[ids].detach()
    # max_attr_prob = max_attr_prob[ids].detach()
    # max_attr_label = max_attr_label[ids].detach()
    # instances.attr_scores = max_attr_prob
    # instances.attr_classes = max_attr_label

    img_bboxes = instances.get_fields()['pred_boxes'].tensor.cpu().numpy()
    roi_features = roi_features.cpu().numpy()

    #print(instances)
    
    return img_bboxes, roi_features


def main():

  args = get_parser()

  with open(arg.img_index_file, 'r') as f:
    ids_map = json.load(f)

  h5file_path = os.path.join(args.save_path, '{}_objects.h5'.format(args.tier))
  object_file = h5py.File(h5file_path,'w')
  object_file.create_dataset("bboxes", (len(ids_map),36,4), dtype='f4')
  object_file.create_dataset("features", (len(ids_map),36,2048), dtype='f4')

  predictor = load_model(args.config_path, args.weight_path)

  for index, image_id in ids_map['image_ix_to_id'].items():
    image_file = os.path.join(args.img_folder, image_id + '.jpg')
    img = cv2.imread(file_path)
    img_bbox, roi_features = doit(img, predictor)
    object_file['bboxes'][index] = img_bbox
    object_file['features'][index] = roi_features
  
  object_file.close()

if __name__ == "__main__":
  main()










