import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from glob import glob
import os


def detect():
    pass


if __name__ == '__main__':
    # Arg setting
    source = r'.\datasets\pano630\val\images\202008250508501754_0950923A.jpg'
    weights = r'.\dentist_cv\exp12\weights\best.pt'
    imgsz = (640, 640)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    max_det = 4  # maximum detections per image
    view_img = True
    save_crop = False  # save cropped prediction boxes
    save_conf = False  # save confidences in --save-txt labels
    line_thickness = 2  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    augment = False  # augmented inference
    visualize = False  # visualize features

    # Model loading
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Data loader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print strin
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            cls_dict = {}
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    cls_dict[int(cls.item())] = xyxy

                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Front tooth numbering
            upper_width = ((cls_dict[1][0] - cls_dict[0][2]) / 4)
            tooth_list = [cls_dict[0], ]
            tooth_number_dict = {
                0: '13',
                1: '12',
                2: '11',
                3: '21',
                4: '22'
            }
            for tooth_number in range(4):
                repeat_bound = 5
                new_xyxy = [
                    tooth_list[tooth_number][2] - repeat_bound,  # new_x = (right_x - left_x) / 4
                    tooth_list[tooth_number][1],
                    tooth_list[tooth_number][2] + upper_width,
                    # new_x2 = (right_x - left_x - right_w) / 4
                    tooth_list[tooth_number][3]
                ]
                tooth_list.append(new_xyxy)

                # FIXME
                # File "E:\Codes\PycharmProjects\dentist-CV\yolov5\utils\plots.py", line 96, in box_label
                #     p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                # ValueError: only one element tensors can be converted to Python scalars
                annotator.box_label(new_xyxy, tooth_number_dict[tooth_number + 1], color=(255, 0, 0))

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


def unlabel_bounding(init_ref, width, labels, repeat_bound=5):
    result = [init_ref, ]
    for tooth_number in range(len(labels)):
        new_xyxy = [
            result[tooth_number][2] - repeat_bound,  # new_x = (right_x - left_x) / 4
            result[tooth_number][1],
            result[tooth_number][2] + width,
            # new_x2 = (right_x - left_x - right_w) / 4
            result[tooth_number][3]
        ]

        result.append({
            'xyxy': new_xyxy,
            'label': labels[tooth_number]
        })

    return result[1:]