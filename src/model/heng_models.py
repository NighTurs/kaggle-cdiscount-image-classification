import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import os
import bson
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from src.heng_cherkeng.net.model.cdiscount.inception_v3 import Inception3
from src.heng_cherkeng.net.model.cdiscount.excited_inception_v3 import SEInception3
from src.data.category_idx import category_to_index_dict

CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180


def read_label_to_category_id(file):
    with open(file, 'r') as file:
        d = eval(file.read())
    return d


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


def doit(net, vecs, ids, dfs, label_to_category_id, category_dict, top_k=10):
    x = Variable(vecs, volatile=True).cuda()
    logits = net(x)
    probs = F.softmax(logits)
    probs = probs.cpu().data.numpy()

    top_k_preds = np.argpartition(probs, -top_k)[:, -top_k:]
    chunk = []
    for i in range(len(ids)):
        product_id = ids[i][0]
        img_idx = ids[i][1]
        for pred_idx in range(top_k):
            chunk.append(
                (product_id, img_idx, category_dict[label_to_category_id[top_k_preds[i, pred_idx]]],
                 probs[i, top_k_preds[i, pred_idx]]))
    chunk_df = pd.DataFrame(chunk, columns=['product_id', 'img_idx', 'category_idx', 'prob'])
    dfs.append(chunk_df)


def model_predict(bson_file, model_name, model_dir, label_to_category_id_file, batch_size, category_idx):
    category_dict = category_to_index_dict(category_idx)

    if model_name == 'inception':
        net = Inception3(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    elif model_name == 'seinception':
        net = SEInception3(in_shape=(3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    else:
        raise ValueError('Unknown model name ' + model_name)

    net.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    net.cuda().eval()

    label_to_category_id = read_label_to_category_id(label_to_category_id_file)

    bson_iter = bson.decode_file_iter(open(bson_file, 'rb'))
    batch_size = batch_size

    dfs = []
    with tqdm() as pbar:
        v = torch.from_numpy(np.zeros((batch_size, 3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), dtype=np.float32))
        ids = []
        for d in bson_iter:
            product_id = d['_id']
            for e, pic in enumerate(d['imgs']):
                image = cv2.imdecode(np.fromstring(pic['picture'], np.uint8), 1)
                x = image_to_tensor_transform(image)
                v[len(ids)] = x
                ids.append((product_id, e))
                if len(ids) == batch_size:
                    doit(net, v, ids, dfs, label_to_category_id, category_dict)
                    pbar.update(len(ids))
                    ids = []
        if len(ids) > 0:
            doit(net, v, ids, dfs, label_to_category_id, category_dict)
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

    args = parser.parse_args()

    category_idx = pd.read_csv(args.category_idx_csv)

    preds = model_predict(args.bson, args.model_name, args.model_dir, args.label_to_category_id_file, args.batch_size,
                          category_idx)
    preds.to_csv(os.path.join(args.model_dir, 'predictions.csv'), index=False)
