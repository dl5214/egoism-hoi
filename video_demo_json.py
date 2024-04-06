# import some common libraries
import json

import numpy as np
import cv2
import random
import os
import torch
from typing import Dict, List
import argparse
import sys

# import some common detectron2 utilities
from detectron2.config import CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, SimpleMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.data.ehoi_dataset_mapper_v1 import *
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.custom_visualizer import *
from detectron2.utils.converters import *


class Args():
    def __init__(self, image_path=None, video_path=None, save_dir="../../output/"):
        self.ref_dataset_json = './data/ref_enigma.json'
        self.weights = './weights/383__33_lf/model_final.pth'  # Update this with the actual path to your weights
        self.cfg_path = None  # Update if you have a specific config path
        self.nms = 0.3
        self.no_cuda = False
        self.seed = 0
        self.cuda_device = 0
        # self.images_path = None  # Update this to process images
        # # self.images_path = '../../data/openmml/tea_img.png'
        # self.video_path = None
        # # self.video_path = '../../data/openmml/tea.mp4'  # Update this to process video
        self.image_path = image_path
        self.video_path = video_path

        # self.save_dir = "../../output/"
        self.save_dir = save_dir
        self.skip_the_fist_frames = 0
        self.duration_of_record_sec = 10000000
        self.hide_depth = False
        self.hide_ehois = False
        self.hide_bbs = False
        self.hide_masks = False
        self.save_masks = False
        self.save_depth_map = False
        self.thresh = 0.5


def format_times(times_dict):
    str_ = ""
    for k, v in times_dict.items():
        str_ += f"\t{k}: {v} ms\t\n"
    return str_


def clear_output(str_):
    for i in range(str_.count('\n') + 1):
        sys.stdout.write("\033[K\033[F")


def load_cfg(args):
    dataset_name = "val_set"
    # Check if the dataset is already registered and remove it if it is
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)

    # Now safe to register the dataset
    register_coco_instances(dataset_name, {}, args.ref_dataset_json, args.ref_dataset_json)
    _ = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    weights_path = args.weights
    cfg_path = os.path.join(weights_path.split("model_")[0], "cfg.yaml") if not args.cfg_path else args.cfg_path
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_path))
    cfg.set_new_allowed(True)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.as_dict()["thing_dataset_id_to_contiguous_id"])
    cfg.MODEL.WEIGHTS = weights_path
    cfg.OUTPUT_DIR = "./output_dir/test"
    cfg.ADDITIONAL_MODULES.NMS_THRESH = args.nms
    cfg.UTILS.VISUALIZER.THRESH_OBJS = args.thresh
    cfg.UTILS.VISUALIZER.DRAW_EHOI = not args.hide_ehois
    cfg.UTILS.VISUALIZER.DRAW_MASK = not args.hide_masks
    cfg.UTILS.VISUALIZER.DRAW_OBJS = not args.hide_bbs
    cfg.UTILS.VISUALIZER.DRAW_DEPTH = not args.hide_depth
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()

    return cfg, metadata


def prepare_labels_and_boxes(output, metadata, frame_time=-1.0):
    # Extract labels and bounding boxes for JSON serialization, now with label names and frame time
    prepared_output = []
    class_names = metadata.get("thing_classes", [])
    for item in output:
        if 'instances' in item:
            instances = item['instances']
            labels = instances.pred_classes.tolist() if instances.has("pred_classes") else []
            boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
            boxes = [[round(coord, 1) for coord in box] for box in boxes]
            for label, box in zip(labels, boxes):
                label_name = class_names[label] if len(class_names) > label else "Unknown"
                prepared_output.append({
                    "time": round(frame_time, 1) if frame_time >= 0 else None,
                    "label": label_name,
                    "box": box
                })
    return prepared_output


def prepare_labels_and_scaled_boxes(original_width, original_height, output, metadata, frame_time=-1.0):
    prepared_output = []
    class_names = metadata.get("thing_classes", [])
    for item in output:
        if 'instances' in item:
            instances = item['instances']
            labels = instances.pred_classes.tolist() if instances.has("pred_classes") else []
            boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
            # Scale the box coordinates to a 0-100 scale based on original dimensions
            # scaled_boxes = [
            #     [
            #         round((coord[0] / original_width) * 100, 1),  # x1
            #         round((coord[1] / original_height) * 100, 1),  # y1
            #         round((coord[2] / original_width) * 100, 1),  # x2
            #         round((coord[3] / original_height) * 100, 1)   # y2
            #     ] for coord in boxes
            # ]
            scaled_boxes = [
                [
                    round((coord[0] / original_width) * 1000),  # x1
                    round((coord[1] / original_height) * 1000),  # y1
                    round((coord[2] / original_width) * 1000),  # x2
                    round((coord[3] / original_height) * 1000)  # y2
                ] for coord in boxes
            ]
            for label, box in zip(labels, scaled_boxes):
                label_name = class_names[label] if len(class_names) > label else "Unknown"
                prepared_output.append({
                    "time": round(frame_time, 1) if frame_time >= 0 else None,
                    "label": label_name,
                    "box": box
                })
    return prepared_output


def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return width, height


def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def main(image_path=None, video_path=None, start_time_sec=0, end_time_sec=None,
         desired_fps=None, save_dir="../../output/"):

    args = Args(image_path, video_path, save_dir)

    json_output_content = []

    kwargs = {}
    kwargs["cuda_device"] = args.cuda_device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg, metadata = load_cfg(args)

    ###INIT MODEL
    converter = MMEhoiNetConverterv1(cfg, metadata)
    model = MMEhoiNetv1(cfg, metadata)

    ####INIT MODEL
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = "cuda:" + str(args.cuda_device) if not args.no_cuda else "cpu"
    model.to(device)
    model.eval()
    print("Modello caricato:", model.device)

    # VISUALIZER AND MAPPER
    visualizer = EhoiVisualizerv1(cfg, metadata, converter, **kwargs)
    mapper = SimpleMapper(cfg)

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():

        ###PROC IMAGES
        if args.image_path:

            width, height = get_image_dimensions(image_path)
            print(f"Width: {width}, Height: {height}")

            #### kwargs init
            kwargs = {}
            kwargs["save_masks"] = args.save_masks
            kwargs["save_depth_map"] = args.save_depth_map
            if args.save_masks:
                os.makedirs(os.path.join(args.save_dir, "masks_processed/"), exist_ok=True)
            if args.save_depth_map:
                os.makedirs(os.path.join(args.save_dir, "depth_maps_processed/"), exist_ok=True)
            save_dir_images = os.path.join(args.save_dir, "images_processed")
            os.makedirs(save_dir_images, exist_ok=True)

            ####IMAGE
            if args.image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(args.image_path)
                r_ = model([mapper(image)])
                # print(f"Inference times:\n {format_times(model._last_inference_times)}")
                # print(type(r_))
                # print(r_)

                # Convert model output to a simplified format containing only labels and bounding boxes
                prepared_output = prepare_labels_and_boxes(r_, metadata)
                print(type(prepared_output))
                print(prepared_output)
                json_output_content.extend(prepared_output)


            elif os.path.isdir(args.image_path):
                n_files = len(os.listdir(args.image_path))
                for id_, file in enumerate(os.listdir(args.image_path)):
                    if args.save_masks:
                        kwargs["save_masks_path"] = os.path.join(args.save_dir,
                                                                 "masks_processed/" + file.split(".")[0] + "_masks.png")
                    if args.save_depth_map:
                        kwargs["save_depth_map_path"] = os.path.join(args.save_dir,
                                                                     "depth_maps_processed/" + file.split(".")[
                                                                         0] + "_depth_map.png")

                    image = cv2.imread(os.path.join(args.image_path, file))
                    r_ = model([mapper(image)])
                    prepared_output = prepare_labels_and_boxes(r_, metadata)
                    print(type(prepared_output))
                    print(prepared_output)


        if args.video_path:

            width, height = get_video_dimensions(video_path)
            print(f"Width: {width}, Height: {height}")

            video_cap = cv2.VideoCapture(args.video_path)
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_interval = int(fps / desired_fps)
            start_frame = start_time_sec * fps
            end_frame = end_time_sec * fps if end_time_sec else total_frames

            video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            while current_frame < end_frame:
                ret, frame = video_cap.read()
                if not ret:
                    break
                if (current_frame - start_frame) % frame_interval == 0:
                    frame_time = current_frame / fps
                    r_ = model([mapper(frame)])
                    prepared_output = prepare_labels_and_scaled_boxes(width, height, r_, metadata, frame_time=frame_time)
                    json_output_content.extend(prepared_output)
                current_frame += 1

            video_cap.release()

    print("Done.")

    # Define the output file path
    output_json_path = os.path.join(args.save_dir, "egoism-hoi_output.json")
    # Serialize and save the json_output_content to a file
    # Open the file for writing
    with open(output_json_path, 'w') as outfile:
        for entry in json_output_content:
            # Serialize each dictionary object to a JSON-formatted string
            json_str = json.dumps(entry)
            # Write the string to the file, followed by a newline character
            outfile.write(json_str + '\n')

    print(f"Output saved to {output_json_path}")

    result = "".join(json.dumps(entry) for entry in json_output_content) + "\n"
    return result


if __name__ == "__main__":
    # main(image_path='../../data/openmml/tea_img.png')
    # start_time_sec and end_time_sec: in seconds
    main(video_path='../../data/openmml/tea.mp4', start_time_sec=10, end_time_sec=20, desired_fps=1)
