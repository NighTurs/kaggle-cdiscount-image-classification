import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import os
import bson
import itertools
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from src.heng_cherkeng.inception_v3 import Inception3
from src.heng_cherkeng.excited_inception_v3 import SEInception3
from src.heng_cherkeng.xception import Xception
from src.heng_cherkeng.resnet101 import ResNet101
from src.data.category_idx import category_to_index_dict

CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180


def read_label_to_category_id(file):
    with open(file, 'r') as file:
        d = eval(file.read())
    return d

def read_train_ids(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    return {int(line) for line in lines}

def pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor


def image_to_tensor_transform(image):
    tensor = pytorch_image_to_tensor_transform(image)
    tensor[0] = tensor[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[1] = tensor[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[2] = tensor[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor


def doit(net, vecs, ids, dfs, label_to_category_id, category_dict, top_k=10, single_prediction=False):
    x = Variable(vecs, volatile=True).cuda()
    logits = net(x)
    preds = F.softmax(logits)
    preds = preds.cpu().data.numpy()

    if single_prediction:
        product_start = 0
        prev_product_id = 0
        chunk = []
        for i, tuple in enumerate(itertools.chain(ids, [(1, 0)])):
            if prev_product_id != 0 and prev_product_id != tuple[0]:
                prods = preds[product_start:i].prod(axis=-2)
                prods = prods / prods.sum()
                top_k_preds = np.argpartition(prods, -top_k)[-top_k:]
                for pred_idx in range(top_k):
                    chunk.append((prev_product_id, 0, top_k_preds[pred_idx], prods[top_k_preds[pred_idx]]))
                product_start = i
            prev_product_id = tuple[0]
    else:
        top_k_preds = np.argpartition(preds, -top_k)[:, -top_k:]
        chunk = []
        for i in range(len(ids)):
            product_id = ids[i][0]
            img_idx = ids[i][1]
            for pred_idx in range(top_k):
                chunk.append(
                    (product_id, img_idx, category_dict[label_to_category_id[top_k_preds[i, pred_idx]]],
                     preds[i, top_k_preds[i, pred_idx]]))
    chunk_df = pd.DataFrame(chunk, columns=['product_id', 'img_idx', 'category_idx', 'prob'])
    dfs.append(chunk_df)


def model_predict(bson_file, model_name, model_dir, label_to_category_id_file, batch_size, category_idx, is_pred_valid,
                  train_ids_file, single_prediction=False):
    category_dict = category_to_index_dict(category_idx)

    if model_name == 'inception':
        net = Inception3(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    elif model_name == 'seinception':
        net = SEInception3(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    elif model_name == 'xception':
        net = Xception(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    elif model_name == 'resnet101':
        net = ResNet101(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    else:
        raise ValueError('Unknown model name ' + model_name)

    net.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    net.cuda().eval()

    label_to_category_id = read_label_to_category_id(label_to_category_id_file)
    if is_pred_valid:
        train_ids = read_train_ids(train_ids_file)

    bson_iter = bson.decode_file_iter(open(bson_file, 'rb'))
    batch_size = batch_size

    dfs = []
    with tqdm() as pbar:
        v = torch.from_numpy(np.zeros((batch_size + 3, 3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), dtype=np.float32))
        ids = []
        for d in bson_iter:
            product_id = d['_id']
            # noinspection PyUnboundLocalVariable
            if is_pred_valid and product_id in train_ids:
                continue
            for e, pic in enumerate(d['imgs']):
                image = cv2.imdecode(np.fromstring(pic['picture'], np.uint8), 1)
                x = image_to_tensor_transform(image)
                v[len(ids)] = x
                ids.append((product_id, e))
            if len(ids) >= batch_size:
                doit(net, v, ids, dfs, label_to_category_id, category_dict, single_prediction)
                pbar.update(len(ids))
                ids = []
        if len(ids) > 0:
            doit(net, v, ids, dfs, label_to_category_id, category_dict, single_prediction)
            pbar.update(len(ids))

    return pd.concat(dfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', required=True, help='Path to bson with products')
    parser.add_argument('--model_name', required=True, help='Model name: inception or seinception')
    parser.add_argument('--model_dir', required=True, help='Output directory for vectors')
    parser.add_argument('--label_to_category_id_file', required=True, help='Hengs label to category mappings file')
    parser.add_argument('--batch_size', type=int, required=False, default=256, help='Batch size')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--predict_valid', action='store_true', required=False, dest='is_predict_valid')
    parser.set_defaults(is_predict_valid=False)
    parser.add_argument('--train_ids_file', required=False, help='Path to Hengs with train ids')
    parser.add_argument('--single_prediction', action='store_true', required=False, dest='single_prediction')
    parser.set_defaults(single_prediction=False)

    args = parser.parse_args()

    category_idx = pd.read_csv(args.category_idx_csv)

    preds = model_predict(args.bson, args.model_name, args.model_dir, args.label_to_category_id_file, args.batch_size,
                          category_idx, args.is_predict_valid, args.train_ids_file, args.single_prediction)
    if args.is_predict_valid:
        if args.single_prediction:
            csv_name = 'valid_single_predictions.csv'
        else:
            csv_name = 'valid_predictions.csv'
    else:
        if args.single_prediction:
            csv_name = 'single_predictions.csv'
        else:
            csv_name = 'predictions.csv'
    preds.to_csv(os.path.join(args.model_dir, csv_name), index=False)

