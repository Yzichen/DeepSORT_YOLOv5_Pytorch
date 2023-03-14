# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import os.path as osp
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
YOLO_ROOT = osp.join(FILE.parents[0], 'yolov5') # YOLOv5 YOLO_ROOT directory
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))  # add YOLO_ROOT to PATH
YOLO_ROOT = Path(os.path.relpath(YOLO_ROOT, Path.cwd()))  # relative
print(YOLO_ROOT)

DeepSort_ROOT = osp.join(FILE.parents[0], 'deep_sort') # YOLOv5 YOLO_ROOT directory
if str(DeepSort_ROOT) not in sys.path:
    sys.path.append(str(DeepSort_ROOT))  # add YOLO_ROOT to PATH
DeepSort_ROOT = Path(os.path.relpath(DeepSort_ROOT, Path.cwd()))  # relative

import torch
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.models.common import DetectMultiBackend

from deep_sort.ds_utils.parser import get_config
from deep_sort.deep_sort import build_tracker
from compute_distance_speed import compute_distance, compute_speed


@smart_inference_mode()
def run(
        weights=YOLO_ROOT / 'yolov5s.pt',  # model path or triton URL
        deepsort_config=DeepSort_ROOT / 'configs/deep_sort.yaml',
        source=YOLO_ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=YOLO_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=YOLO_ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        draw_trajectory=False,
        show_distance=True,
        show_speed=True,
        compute_interval=5,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)    # 根据source的后缀来确认source是否是一张图片或是视频.
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # initialize DeepSORT
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)

    use_cuda = device.type != 'cpu' and torch.cuda.is_available()
    deepsort = build_tracker(cfg, use_cuda=use_cuda)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    pre_count = 0
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        # path: 文件路径
        # im: (3, H, W)
        # im0: (ori_H, ori_W, 3)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim   # (1, 3, H, W)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            # pred: Tuple(
            #     z: (B=1, N_anchor*H3*H4 + N_anchor*H4*W4 + N_anchor*H5*W5, 5+n_cls)   5: cx, cy, w, h, obj
            #     x: List[(B=1, N_anchor, H3, W3, N_cls+5), (B=1, N_anchor, H4, W4, N_cls+5), ...]
            # )
        # NMS
        with dt[2]:
            # pred: List[(N_pred0, 6), ]  6: x1, y1, x2, y2, conf, cls
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 如果读取了下一个视频，则需要重置跟踪器.
        count = dataset.count
        if count != pre_count:
            # deepsort.tracker.tracks = []
            # deepsort.tracker._next_id = 1
            deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # det: (N_pred, 6)  6: x1, y1, x2, y2, conf, cls
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)     # p: 图像路径; im0: (ori_H, ori_W, 3); frame: 对于图像恒为0, 对于视频，表示视频中的第几帧.

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # (4, )  4: [Ori_W, Ori_H, Ori_W, Ori_H]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size   (N_pred, 6)  此时x1, y1, x2, y2已经scale到原图的尺寸.
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results  统计该图像中的目标（4 persons, 1 bus）
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                with dt[3]:
                    # ****************************** deepsort ****************************
                    bbox_xywh = xyxy2xywh(det[:, :4]).cpu()    # (N_pred, 4), (cx, cy, w, h)
                    confs = det[:, 4:5].cpu()   # (N_pred, 1)   conf
                    clss = det[:, 5:6].cpu()    # (N_pred, 1)
                    outputs = deepsort.update(bbox_xywh, confs, clss, im0)    # (#ID, 6)  x1,y1,x2,y2, cls, track_ID

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if save_txt:  # Write to file
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #         with open(f'{txt_path}.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
                #
                #     if save_img or save_crop or view_img:  # Add bbox to image
                #         c = int(cls)  # integer class
                #         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                #         annotator.box_label(xyxy, label, color=colors(c, True))
                #     if save_crop:
                #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Write results
                for *xyxy, cls, track_ID in reversed(outputs):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 先初始化label.
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} Id:{track_ID}')

                        # 绘制跟踪轨迹
                        if draw_trajectory:
                            track_index = deepsort.tracker.active_targets.index(track_ID)
                            traces = [xywh[0:2] for xywh in deepsort.tracker.tracks[track_index].traces]
                            points = np.stack(traces, axis=0)
                            annotator.line(points, color=colors(track_ID, True))

                        # 计算距离
                        if show_distance:
                            if frame % compute_interval == 0:
                                # 每compute_interval计算一次，否则一直在变化, 可视化效果不好.
                                bottom_center = np.array(((xyxy[0]+xyxy[2])/2, xyxy[3]))     # (cx, y2)
                                distance = compute_distance(bottom_center)
                                # 记录在track中
                                track_index = deepsort.tracker.active_targets.index(track_ID)
                                deepsort.tracker.tracks[track_index].distance = distance
                            else:
                                # 对于已经计算过距离的，直接取出即可
                                track_index = deepsort.tracker.active_targets.index(track_ID)
                                if hasattr(deepsort.tracker.tracks[track_index], 'distance'):
                                    distance = deepsort.tracker.tracks[track_index].distance
                                else:   # 没计算过的，需要计算.
                                    bottom_center = np.array(((xyxy[0]+xyxy[2])/2, xyxy[3]))   # (cx, y2)
                                    distance = compute_distance(bottom_center)
                                    # 记录在track中
                                    track_index = deepsort.tracker.active_targets.index(track_ID)
                                    deepsort.tracker.tracks[track_index].distance = distance
                            # 更新label, 显示距离.
                            label += f'\nDis:{distance:.2f}m '

                        if show_speed:
                            if frame % compute_interval == 0:
                                track_index = deepsort.tracker.active_targets.index(track_ID)
                                traces = [(xywh[0], xywh[1]+xywh[3]/2) for xywh in deepsort.tracker.tracks[track_index].traces]     # (cx, y2)
                                traces_points = np.array(traces, np.float32)   # (N, 2)
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                vx, vy, v = compute_speed(traces_points, fps=fps, interval=5)
                                # 记录在track中
                                track_index = deepsort.tracker.active_targets.index(track_ID)
                                deepsort.tracker.tracks[track_index].vx = vx
                                deepsort.tracker.tracks[track_index].vy = vy
                                deepsort.tracker.tracks[track_index].v = v

                            else:
                                if hasattr(deepsort.tracker.tracks[track_index], 'v'):
                                    v = deepsort.tracker.tracks[track_index].v
                                    vx = deepsort.tracker.tracks[track_index].vx
                                    vy = deepsort.tracker.tracks[track_index].vy
                                else:
                                    track_index = deepsort.tracker.active_targets.index(track_ID)
                                    traces = [(xywh[0], xywh[1] + xywh[3] / 2) for xywh in
                                              deepsort.tracker.tracks[track_index].traces]  # (cx, y2)
                                    traces_points = np.array(traces, np.float32)   # (N, 2)
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    vx, vy, v = compute_speed(traces_points, fps=fps, interval=5)
                                    # 记录在track中
                                    track_index = deepsort.tracker.active_targets.index(track_ID)
                                    deepsort.tracker.tracks[track_index].vx = vx
                                    deepsort.tracker.tracks[track_index].vy = vy
                                    deepsort.tracker.tracks[track_index].v = v
                            # 更新label, 显示速度.
                            label += f'\nVx: {vx:.2f}km/h\nVy: {vy:.2f}km/h\nV: {v:.2f}km/h'

                        # 绘制检测框
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = [cls, *xywh, track_index]  # label format
                        if show_distance:
                            line.append(distance)
                        if show_speed:
                            line.append(vx, vy, v)
                        line = tuple(line)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%.3g ' * len(line)).rstrip() % line + '\n')

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}, detection: {dt[1].dt * 1E3:.1f}ms,  "
                    f"track: {dt[3].dt * 1E3:.1f}ms",)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms Track per image at shape,  {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=YOLO_ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument("--deepsort_config", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--source', type=str, default=YOLO_ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=YOLO_ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=YOLO_ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--draw-trajectory', action='store_true')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

