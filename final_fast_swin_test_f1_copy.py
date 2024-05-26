from dataset.fast.fast_tt import tt_test_data_dir
from dataset.fast.fast_ic15 import ic15_test_data_dir
from dataset.fast.fast_sample_data import sample_test_data_dir, sample_test_gt_dir

from collections import defaultdict
import pdb
from dataset.utils import get_img
from lpn_run_copy import predict_languages_in_folder, predict_language

from utils import ResultFormat
import subprocess
import argparse

# swin
import os
import cv2
import mmcv
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
import torch
import datetime
from pathlib import Path
from swin_utils import load_setting, load_tokenizer
from swin_models import SwinTransformerOCR
from swin_dataset import CustomCollate

# fast_test
import argparse
import sys
from mmcv import Config
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from mmcv.cnn import get_model_complexity_info
import logging
import warnings
warnings.filterwarnings('ignore')

import json
import pdb

ctw_pred_dir = 'outputs/submit_ctw/'
tt_pred_dir = 'outputs/submit_tt/'
ic15_pred_dir = 'outputs/submit_ic15/'
sample_pred_dir = 'outputs/submit_sample/'

rand_r = random.randint(100, 255)
rand_g = random.randint(100, 255)
rand_b = random.randint(100, 255)


def test(test_loader, model, cfg):
    
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)
    results  = dict()

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
        logging.info('Testing %d/%d\r' % (idx, len(test_loader)))
        # prepare input
        if not args.cpu:
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        data.update(dict(cfg=cfg))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        # save result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            rf.write_result(image_name, outputs['results'][index])
            results[image_name] = outputs['results'][index]

    
    results = json.dumps(results)
    
    # json 파일에 Detection 결과 저장
    with open('outputs/output.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False)
        print("write json file success!")
        
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def main(args):
    cfg = Config.fromfile(args.fast_config)

    if args.min_score is not None:
        cfg.test_cfg.min_score = args.min_score
    if args.min_area is not None:
        cfg.test_cfg.min_area = args.min_area

    cfg.batch_size = args.batch_size

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker,
        pin_memory=False
    )
    # model
    model = build_model(cfg.model)
    
    if not args.cpu:
        model = model.cuda()
    
    if args.fast_checkpoint is not None:
        if os.path.isfile(args.fast_checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.fast_checkpoint))
            logging.info("Loading model and optimizer from checkpoint '{}'".format(args.fast_checkpoint))
            sys.stdout.flush()
            checkpoint = torch.load(args.fast_checkpoint)
            
            if not args.ema:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint['ema']

            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.fast_checkpoint))
            raise
    
    model = rep_model_convert(model)

    # fuse conv and bn
    model = fuse_module(model)
    
    if args.print_model:
        model_structure(model)
    

    # Detect
    model.eval()
    test(test_loader, model, cfg)

def get_pred(pred_path):
    lines = mmcv.list_from_file(pred_path)
    bboxes = []
    cropnames = []
    base_filename, _ = os.path.splitext(os.path.basename(pred_path))
    
    filename = base_filename.replace("res_", "")
    
    for i, line in enumerate(lines):
        line = line.encode('utf-8').decode('utf-8-sig').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        bbox = [int(gt[i]) for i in range(len(gt))]
        bboxes.append(bbox)
        cropname = base_filename.replace("res_", "") + '_' + str(i)
        cropnames.append(cropname)
        
    result = {
        "Image": filename,
        "crop_name" : cropnames,
        "bbox": bboxes,
        "text": []
        }

    return np.array(bboxes), result

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def draw(img, crop_name, boxes, dataset, index, model, tokenizers, cfg): # 크롭 이미지들 LPN, FAST
    
    rand_r = random.randint(100, 255)
    rand_g = random.randint(100, 255)
    rand_b = random.randint(100, 255)
    
    predicted_languages ={}
    
    cropped_img = None
    prediction = None
    
    SEED = 0
    prediction_dict = []
    img_copy = img.copy()
    # pdb.set_trace()

    imgs = torch.Tensor([]).to('cuda')
    for i, box in enumerate(boxes):
        
        if dataset and index is not None: # pred 일 때만 
            
            box = [abs(val) for val in box]
            
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # #bird-eye-view test
            # 좌표 정렬
            sorted_by_x = sorted(pts, key=lambda coord: coord[0, 0])
            left_sorted = sorted(sorted_by_x[:2], key=lambda coord: coord[0, 1])
            right_sorted = sorted(sorted_by_x[2:], key=lambda coord: coord[0, 1])
            pts_sorted = np.vstack((left_sorted, right_sorted))
            
            pts[0][0] = pts_sorted[0][0]
            pts[1][0] = pts_sorted[2][0]
            pts[2][0] = pts_sorted[3][0]
            pts[3][0] = pts_sorted[1][0]
            
            # 가로 길이 계산
            width1 = abs(pts_sorted[2][0][0] - pts_sorted[0][0][0])
            width2 = abs(pts_sorted[1][0][0] - pts_sorted[3][0][0])
            max_width = max(width1, width2)

            # 세로 길이 계산
            height1 = abs(pts_sorted[1][0][1] - pts_sorted[0][0][1])
            height2 = abs(pts_sorted[3][0][1] - pts_sorted[2][0][1])
            max_height = max(height1, height2)
            
            pts_dst = np.array([
                [0, 0],
                [max_width, 0],
                [max_width, max_height],
                [0, max_height]
            ], dtype=np.float32)
            
            pts_src = pts.reshape((4, 2)).astype(np.float32)
            
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

            cropped_img = cv2.warpPerspective(img_copy, matrix, (int(max_width), int(max_height)))

            if cropped_img.shape[-1] == 0:
                continue
            elif cropped_img.shape[-1] == 4:
                cropped_img = cropped_img[:, :, :3]

            collate = CustomCollate(cfg)
            x = collate.ready_image(cropped_img).to('cuda')
            imgs = torch.cat([imgs, x], dim=0)

    return imgs

def draw_text_on_image(img, text, position, font_path):

    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    color = (rand_r, rand_g, rand_b)

    font_scale = 0.015
    image_width, image_height = img.shape[:2]
    font_size = int(image_height * font_scale)
    font_pil = ImageFont.truetype(font_path, font_size)
    
    draw = ImageDraw.Draw(image_pil)
    outline_thickness = 3
    
    draw.text(position, text, font=font_pil, fill=color, stroke_width=outline_thickness, stroke_fill='black')
    
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

batch = 100

def Acc_MemAndImgs(crops_list, mems_list):
    for lan, crops, mems in zip(['En', 'Ko', 'Ja', 'Ch'], crops_list, mems_list):
        if crops.shape[0] != len(mems):
            print(f'{lan} Size Error Crops: {crops.shape[0]}, Mems: {len(mems)}')
        if crops.shape[0] >= batch:
            print(f'{lan} Size Not Input on Swin: {crops.shape[0]}, Mems: {len(mems)}')

def calculate_f1(confusion_list):
    TP, FN, FP = confusion_list

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0    

    return TP, FN, FP, f1_score

def visual(data_dir, gt_dir, pred_dir, dataset, models, tokenizers):
    # pdb.set_trace()
    model = models['Korean']
    
    img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.PNG')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.JPG')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.jpeg')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.JPEG')])
    
    img_paths, pred_paths = [], []
    
    
    for idx, img_name in enumerate(img_names):
        img_path = data_dir + img_name
        img_paths.append(img_path)
        
        # collect paths of ground truths and predictions
        if dataset == 'ctw': # CTW-1500
            pred_name = img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'tt': # Total-Text
            pred_name = img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'ic15': # ICDAR 2015
            pred_name = "res_" + img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'Sample': # ICDAR 2015
            pred_name = "res_" + img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
            
        # collate = None
        # predicted_languages = {}
        font_path = '/home/pirl/Desktop/OCR/FAST/SourceHanSansK-Regular.otf'

    # pdb.set_trace()

    E_Crops, K_Crops, J_Crops, C_Crops = torch.Tensor([]).to('cuda'), torch.Tensor([]).to('cuda'), torch.Tensor([]).to('cuda'), torch.Tensor([]).to('cuda')
    E_Model, K_Model, J_Model, C_Model = models['Latin'].to('cuda'), models['Korean'].to('cuda'), models['Japanese'].to('cuda'), models['Chinese'].to('cuda')
    E_mem, K_mem, J_mem, C_mem = [[] for _ in range(4)]

    gts_info = defaultdict(dict)
    preds_info = defaultdict(dict)
    imgs_path = {}

    total_cnt = 0

    # 이미지를 한번에 여러개 넣어서 한번에 예측할 것임
    for index, (img_path, pred_path) in tqdm(enumerate(zip(img_paths, pred_paths)), total = len(img_paths), desc = 'Processing Images', dynamic_ncols = True, mininterval=0.5):
        img = get_img(img_path) # load image
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        imgs_path[img_name] = img_path

        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print(time)
        
        # 한 이미지 예측한 결과 -> 크롭 이미지 (1, 2, 3) 딕셔너리 형태
        _, result_dict = get_pred(pred_path)

        # pdb.set_trace()

        # 여기도 수정 언어별로 따로 저장해야함 여기 빈 이미지도 있나봄
        crop_imgs = draw(img, img_name, result_dict['bbox'], dataset, index, model, tokenizers, cfg)

        if crop_imgs.shape[0] == 0:
            continue

        # GT Information -> to dictionary
        if result_dict['Image'] == img_name:
            with open(gt_dir + img_name + '.txt', 'r', encoding='utf-8-sig') as f:
                labels = f.readlines()

            for i, line in enumerate(labels):
                parts = line.split(',')
                coords = [float(x) for x in parts[:8]]
                text = parts[-1].strip()
                # Calculate center coordinates for ground truth
                gt_center_x = sum(coords[::2]) / 4
                gt_center_y = sum(coords[1::2]) / 4
                gt_center_coords = (gt_center_x, gt_center_y)

                gts_info[img_name][i] = (coords, text, gt_center_coords)

        # Pred Information -> to dictionary(if out of memory, then evaluation)
        E_idx, K_idx, J_idx, C_idx = [[] for _ in range(4)]
        pred_lan = predict_language(crop_imgs)
        for i, lan in enumerate(pred_lan): # ['한국', '라틴', ..., '중국']
            if lan == 'Latin': E_idx.append(i)
            elif lan == 'Korean': K_idx.append(i)
            elif lan == 'Japanese': J_idx.append(i)
            elif lan == 'Chinese': C_idx.append(i)

        # pdb.set_trace()

        Crops_list = [E_Crops, K_Crops, J_Crops, C_Crops]
        lans_mem = [E_mem, K_mem, J_mem, C_mem]

        for i, (indices, mem) in enumerate(zip([E_idx, K_idx, J_idx, C_idx], lans_mem)):
            Crops_list[i] = torch.cat([Crops_list[i], crop_imgs[indices]], dim=0)
            mem.extend(list(map(lambda x: [img_name, x, result_dict['bbox'][x]], indices)))

        E_Crops, K_Crops, J_Crops, C_Crops = Crops_list # 크롭 이미지들 결합 후 저장

        # pdb.set_trace()

        for i, (lan_model) in enumerate([E_Model, K_Model, J_Model, C_Model]):
            if Crops_list[i].shape[0] >= batch or index >= len(img_paths)-1: # 배치사이즈보다 메모리가 넘거나 마지막이면
                out_size = min(batch, len(lans_mem[i]))
                total_cnt += out_size
                test_imgs = Crops_list[i][:out_size, :, :, :]
                test_mem = lans_mem[i][:out_size]

                lan_model.eval()
                swin_predictions = lan_model.predict(test_imgs) #['항', '나안' ...]
                for j in range(out_size):
                    img_name, crop_idx, bbox, text = *test_mem[j], swin_predictions[j]
                    center_x = sum(bbox[::2]) / 4
                    center_y = sum(bbox[1::2]) / 4
                    pred_center_coords = (center_x, center_y)
                    preds_info[img_name][crop_idx] = (bbox, text, pred_center_coords)

                # 앞에 데이터 삭제 
                Crops_list[i] = Crops_list[i][out_size:, :, :, :]
                lans_mem[i] = lans_mem[i][out_size:]

                # dict에 넣어야할 정보 bbox, swin_prediction, center_coords

        E_Crops, K_Crops, J_Crops, C_Crops = Crops_list # 크롭 이미지들 Swin 학습 후 변형 정보
        E_mem, K_mem, J_mem, C_mem = lans_mem

        Acc_MemAndImgs(Crops_list, lans_mem)
                
    confusion_list = [0, 0, 0]  # TP, FN, FP
    save_img = True
    for img_name in tqdm(gts_info.keys()):
        gt_list = [0]* len(gts_info[img_name].keys())
        
        for gt_crop_idx, gt_item in gts_info[img_name].items():
            gt_bbox, gt_text, gt_center_coords = gt_item
            pred_pair_iou = []

            for pred_crop_idx, pred_item in preds_info[img_name].items():
                pred_bbox, pred_text, pred_center_coords = pred_item
                iou = compute_iou(pred_bbox, gt_bbox)
                pred_pair_iou.append([pred_crop_idx, pred_text, iou])
 
            if not pred_pair_iou:
                continue

            pred_pair_iou.sort(key=lambda x: x[-1], reverse=True)

            if (pred_pair_iou[0][-1]>0.5) and (pred_pair_iou[0][1] == gt_text):
                gt_list[gt_crop_idx] = 1

        confusion_list[0] += gt_list.count(1) # TP
        confusion_list[1] += gt_list.count(0) # FN
        confusion_list[2] += (len(preds_info[img_name].keys())-gt_list.count(1)) # FP

        TP, FN, FP, f1_score = calculate_f1(confusion_list)

        tqdm._instances.clear()
        print(img_name, f"누적 TP : {TP}, FP : {FP}, FN : {FN}, f1-score : {round(f1_score, 4)} ", flush=True)

        if save_img:
            import numpy as np

            save_folder = f"visual/{dataset}"
            mmcv.mkdir_or_exist(save_folder)
            
            img = get_img(imgs_path[img_name])
            for _, pred_item in preds_info[img_name].items():
                pred_bbox, pred_text, _ = pred_item
                pred_bbox = np.array(pred_bbox, np.int32).reshape(4, 2)
                cv2.polylines(img, [pred_bbox], isClosed=True, color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), thickness=10)
                img = draw_text_on_image(img, pred_text, (pred_bbox[0][0], pred_bbox[0][1]-30), font_path)
            
            h, w, _ = img.shape
            img = cv2.resize(img, (w//3, h//3), cv2.INTER_AREA)
            cv2.imwrite(f"{save_folder}/{img_name}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # save images into visual/ 
    
    return calculate_f1(confusion_list)[-1]

def compute_iou(pred_box, gt_box):
    pred_x1 = min(pred_box[::2])
    pred_y1 = min(pred_box[1::2])
    pred_x2 = max(pred_box[::2])
    pred_y2 = max(pred_box[1::2])

    gt_x1 = min(gt_box[::2])
    gt_y1 = min(gt_box[1::2])
    gt_x2 = max(gt_box[::2])
    gt_y2 = max(gt_box[1::2])
    
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)

    pred_area = abs((pred_x2-pred_x1)*(pred_y2-pred_y1))
    gt_area = abs((gt_x2-gt_x1)*(gt_y2-gt_y1))

    return inter_area / (pred_area + gt_area - inter_area + 1e-7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    # fast_test parser
    
    parser.add_argument('fast_config', help='config file path')
    parser.add_argument('fast_checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--worker', default=16, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    
    parser.add_argument('--dataset', nargs='?', type=str, required=True,
                        choices=['tt', 'ctw', 'ic15','Sample'])
    parser.add_argument('--show-gt', action="store_true")
    
    # swin parser
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",help="Experiment settings")
    parser.add_argument("--tokenizer", "-tk", type=str, required=True, help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Load model weight in checkpoint")
    

    # show the ground truths with predictions
    args = parser.parse_args()
    
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)
    
    
    main(args)
    
    # Detection 결과 output/output.json에 있음
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = args.tokenizer.split(',')
    checkpoint = args.checkpoint.split(',')
    
    tokenizers = {
        "Korean": load_tokenizer(tokenizer[0]),
        "Latin": load_tokenizer(tokenizer[1]),
        "Chinese": load_tokenizer(tokenizer[2]),
        "Japanese": load_tokenizer(tokenizer[3]),
    }

    models = {
        "Korean": SwinTransformerOCR(cfg, tokenizers["Korean"]),
        "Latin": SwinTransformerOCR(cfg, tokenizers["Latin"]),
        "Chinese": SwinTransformerOCR(cfg, tokenizers["Chinese"]),
        "Japanese": SwinTransformerOCR(cfg, tokenizers["Japanese"]),
    }

    checkpoint_paths = {
        "Korean": checkpoint[0],
        "Latin": checkpoint[1],
        "Chinese": checkpoint[2],
        "Japanese": checkpoint[3],
    }
    
    for lang, model in models.items():
        saved = torch.load(checkpoint_paths[lang], map_location=device)
        model.load_state_dict(saved['state_dict'])
    
    thickness = {'ctw':4, 'tt':4, 'ic15': 4, 'Sample' : 4}
    
    if args.dataset == 'tt':
        test_data_dir = tt_test_data_dir
        pred_dir = tt_pred_dir
    elif args.dataset == 'ic15':
        test_data_dir = ic15_test_data_dir
        pred_dir = ic15_pred_dir
    elif args.dataset == 'Sample':
        test_data_dir = sample_test_data_dir
        pred_dir = sample_pred_dir
        test_gt_dir = sample_test_gt_dir
    
    # pdb.set_trace()
    # print(test_data_dir)
    f1_score = visual(test_data_dir, test_gt_dir, pred_dir, args.dataset, models, tokenizers)
    
    # time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print(time)
    # tp = fp = fn = 0
    
    # tp += TP
    # fp += FP
    # fn += FN
    
    # precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    # recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    print(f"f1_score : {f1_score:.4f}")
    