import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours

from matplotlib import patches,  lines
from matplotlib.patches import Polygon

import cv2

from sort.sort import Sort as SortTracker
import os
from matplotlib import cm
import shutil

torch.set_grad_enabled(False)


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_results(pil_img, scores, boxes, labels, masks=None):
    plt.figure(figsize=(16, 10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
        masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
            continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=c)
            ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    plt.show()


model_pc = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_phrasecut', pretrained=True, return_postprocessor=False)
model_pc = model_pc.cuda()
model_pc.eval()


def plot_inference_segmentation(im, caption):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0).cuda()

  # propagate through the model
  outputs = model_pc(img, [caption])

  # keep only predictions with 0.9+ confidence
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  keep = (probas > 0.9).cpu()

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

  tokenized = model_pc.detr.transformer.tokenizer.batch_encode_plus([caption], padding="longest", return_tensors="pt").to(img.device)

  # Extract the text spans predicted by each box
  positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
  predicted_spans = defaultdict(str)
  for tok in positive_tokens:
    item, pos = tok
    if pos < 255:
        span = tokenized.token_to_chars(0, pos)
        predicted_spans [item] += " " + caption[span.start:span.end]

  # labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]

  return {"bboxes": bboxes_scaled,
          # "labels": labels,
          "confs": probas[keep]}


def visualize(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i]

        x0, y0, x1, y1, id = map(int, box[:5])

        color_ = _COLORS[id%_COLORS.shape[0]]
        color = (color_ * 255).astype(np.uint8).tolist()
        text = '%d'%(id)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 1.5, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 5)

        txt_bk_color = (color_ * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 1.5, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def track(filename, output, request, max_age=3, min_hits=3, iou_threshold=0.3, log=False):
    cap = cv2.VideoCapture(filename)
    fps = int(cap.get(5))
    size = (int(cap.get(3)), int(cap.get(4)))

    mot_tracker = SortTracker(max_age=max_age,
                              min_hits=min_hits,
                              iou_threshold=iou_threshold)  # create instance of the SORT tracker
    # dirname = "answ"
    # try:
    #   shutil.rmtree(dirname)
    # except Exception as e:
    #   print(e)
    # os.mkdir(dirname)
    frame_num = 0
    if log:
        file_log = open("log.txt", 'w')

    # all_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # out = display(progress(0, all_frames), display_id=True)

    out_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while True:
        _, im = cap.read()
        if im is None:
            break
        pil_image = Image.fromarray(np.uint8(im[:, :, ::-1])).convert('RGB')
        detection = plot_inference_segmentation(pil_image, request)
        bboxes, confs = detection["bboxes"].numpy(), detection["confs"].numpy()
        if len(confs) == 0:
            new_im = im
        else:
            detects = np.c_[bboxes, np.reshape(confs, (-1, 1))]
            # print(frame_num, detects)

            trackers = mot_tracker.update(detects)

            # print(trackers)
            if log:
                for ind in range(trackers.shape[0]):
                    id = int(trackers[ind][4])
                    file_log.write(
                        f"{frame_num + 1},{id},{trackers[ind][0]},{trackers[ind][1]},{trackers[ind][2] - trackers[ind][0]},{trackers[ind][3] - trackers[ind][1]},1,-1,-1,-1\n")

            new_im = visualize(im, trackers)

        # cv2.imwrite(f"./{dirname}/{frame_num:06}.jpg", new_im)
        out_video.write(new_im)

        frame_num += 1
        # out.update(progress(frame_num, all_frames))

    cap.release()
    if log:
        file_log.close()

    out_video.release()
