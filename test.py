import torch
import os
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist,Dataset_test
import glob
from models import BFINet
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
from torch import nn

def build_model(model_type):

    if model_type == "field":
        model = BFINet(path=r'E:\new_parcel_model\New_0415\preweight\pvt_v2_b2.pth')

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()

    args.model_file = './weight/100.pt'
    args.save_path = r'E:\new_parcel_model\HuN\1'
    args.model_type = 'field'
    args.test_path = r'E:\new_parcel_model\HuN\1\image'


    test_path = os.path.join(args.test_path, "*.tif")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    test_file_names = glob.glob(test_path)
    valLoader = DataLoader(Dataset_test(test_file_names), batch_size=1)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    # model = nn.DataParallel(model)  #
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2 = model(inputs)

        ##
        # outputs4, outputs5 = model(torch.flip(inputs, [-1]))
        # predict_2 = torch.flip(outputs4, [-1])
        # outputs7, outputs8 = model(torch.flip(inputs, [-2]))
        # predict_3 = torch.flip(outputs7, [-2])
        # outputs10, outputs11 = model(torch.flip(inputs, [-1, -2]))
        # predict_4 = torch.flip(outputs10, [-1, -2])
        # predict_list = outputs1 + predict_2 + predict_3 + predict_4
        # pred1 = predict_list/4.0

        outputs1 = torch.sigmoid(outputs1)
        outputs1 = outputs1.detach().cpu().numpy().squeeze()

        res = np.zeros((256, 256))
        res[outputs1>0.5] = 255
        res[outputs1<=0.5] = 0

        res = np.array(res, dtype='uint8')
        output_path = os.path.join(
            save_path, os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)