import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
from losses import *
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


def pixel_eval(pred_img, actual_img):
    # pred_img = gdal.Open(pred_path)
    y_pred = pred_img

    # actual_img = gdal.Open(true_path)
    y_true = actual_img

    # 将tif影像转换为二进制掩码
    y_pred = (y_pred == 255).astype(np.int8)
    y_true = (y_true == 255).astype(np.int8)

    # 假设 y_true 和 y_pred 已经准备好
    # precision = precision_score(y_true, y_pred, average='weighted')  # 对于多分类问题，可以指定'weighted'平均方式
    # recall = recall_score(y_true, y_pred, average='weighted')
    # f1 = f1_score(y_true, y_pred, average='weighted')

    # 如果是二分类问题，可以省略average参数或者设置为'micro'或'macro'
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    mcc = matthews_corrcoef(y_true.reshape(-1), y_pred.reshape(-1))

    return precision, recall, f1, mcc



def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            crition = BCEDiceLoss()
            loss  = crition(outputs[0],targets)
            # loss = F.nll_loss(outputs[0], targets.squeeze(1))
            losses.append(loss.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    # f1_score = []
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)
            mask_out = torch.sigmoid(outputs[0])
            output_mask = mask_out.detach().cpu().numpy().squeeze()
            output_mask[output_mask>0.5]= 255
            output_mask[output_mask <=0.5] = 0

            # precision, recall, f1, mcc = pixel_eval(output_mask, (targets*255).detach().cpu().numpy().squeeze())
            # f1_score.append(f1)
            # output_final = np.argmax(output_mask, axis=1).astype(float)
            output_final = torch.from_numpy(output_mask).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train",val_batch_size)
                save_image(output_final, "Prediction_train",val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)
            break

    # return np.mean(f1_score)


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img tif files")
    parser.add_argument("--val_path", type=str, help="path to img tif files")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_contour",
        help="select distance transform type - dist_mask,dist_contour,dist_contour_tif",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser























