import warnings
# from thop import profile
warnings.filterwarnings('ignore')
import glob
import logging
import os
import random
import torch
from dataset import DatasetImageMaskContourDist
from losses import LossF, awl, weighted_bce, BCEDiceLoss
from models import BFINet
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import visualize, create_train_arg_parser,evaluate
# from torchsummary import summary
from sklearn.model_selection import train_test_split
# from torchvision.transforms import functional as F
# import numpy as np
###
def define_loss(loss_type, weights=[1, 1]):

    if loss_type == "field":
        criterion = LossF(weights)

    return criterion


def build_model(model_type):

    if model_type == "field":
        model = BFINet(path=r'E:\new_parcel_model\New_0415\preweight\pvt_v2_b2.pth')

    return model


def train_model(model, targets, model_type, criterion1,  criterion2, optimizer, optimizer1):

    if model_type == "field":

        optimizer.zero_grad()
        optimizer1.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss1 = criterion1(outputs[0], targets[0])
            loss2 = criterion2(outputs[1], targets[1])
            # loss = criterion(
            #     outputs[0], outputs[1], targets[0], targets[1]
            # )
            loss = awl(loss1, loss2)
            loss.backward()
            optimizer.step()
            optimizer1.step()

    return loss


if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()
    # args.pretrained_model_path = r'G:\save\50.pt'

    args.train_path = r'E:\new_parcel_model\HuN\train\image'
    args.model_type = 'field'
    args.save_path = r'E:\new_parcel_model\New\weight'

    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")

    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"))
    random.shuffle(train_file_names)

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    print(device)
    model = build_model(args.model_type)

    # if torch.cuda.device_count() > 0:           #
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model = model.to(device)

    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)
    print('train',args.use_pretrained)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,train_file),
        batch_size=args.batch_size,drop_last=False,  shuffle=True
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file),drop_last=False,
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file),
        batch_size=args.val_batch_size,drop_last=False, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer1 = torch.optim.Adam(awl.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, int(1e10), eta_min=1e-5) ##
    criterion1 = BCEDiceLoss()
    criterion2 = weighted_bce()

    best_f1 = 0.0
    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i, (img_file_name, inputs, targets1, targets2) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)

            targets = [targets1, targets2]

            loss = train_model(model, targets, args.model_type, criterion1, criterion2, optimizer, optimizer1)

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        scheduler1.step()

        epoch_loss = running_loss / len(train_file_names)
        print(epoch_loss)

        if epoch % 1 == 0:

            dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar("loss_valid", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss, dev_loss))
            # print("F1 Score: {:.3f}".format(f1))
        else:
            print("Global Loss:{} ".format(epoch_loss))

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        # if f1>best_f1:
        #     best_f1 = f1
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )


