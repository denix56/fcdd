import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from matplotlib.pyplot import get_cmap
import numpy as np
import cv2 as cv
from sklearn.metrics import roc_auc_score

class TBLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, loss, lr, epoch):
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('LR', lr, epoch)
        self.writer.flush()

    def add_scalars(self, loss, loss_normal, loss_anomal, lr, epoch, train=True):
        tag = 'train' if train else 'test'
        self.writer.add_scalars('Loss', {
            'total_{}'.format(tag): loss,
            'normal_{}'.format(tag): loss_normal,
            'anomalous_{}'.format(tag): loss_anomal
        }, epoch)

        if lr is not None:
            self.writer.add_scalar('LR', lr, epoch)
        self.writer.flush()

    def add_roc_auc_score(self, y_true, y_score, epoch: int):
        auc = roc_auc_score(y_true, y_score)
        self.writer.add_scalar('test/roc_auc_score', auc, global_step=epoch)
        self.writer.flush()

    def add_images(self, inputs: torch.Tensor, anomaly_score: torch.Tensor, gt_maps: (None, torch.Tensor), outputs: torch.Tensor,
                   labels: torch.Tensor, epoch: int, scale_each: bool = False):
        #main_tag = 'normal' if normal else 'anomalous'

        cmap = get_cmap('jet')
        outputs_new = []
        outputs = outputs.clone()

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        if scale_each:
            for t in outputs:
                norm_ip(t.abs_(), float(t.min()), float(t.max()))
        else:
            norm_ip(outputs.abs_(), float(outputs.min()), float(outputs.max()))

        for img in outputs.squeeze(dim=1):
            outputs_new.append(cmap(img.detach().cpu().numpy())[..., :3])
        alpha = 0.5
        outputs = torch.tensor(outputs_new).permute(0, 3, 1, 2)
        outputs = inputs.cpu() * (1 - alpha) + outputs * alpha

        inputs_np = inputs.permute(0, 2, 3, 1).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy().copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (40, 500)
        fontScale = 1
        fontColor = (255, 0, 0)
        thickness = 2
        lineType = cv.LINE_AA

        for i in range(inputs_np.shape[0]):
            cv.putText(inputs_np[i], '{:.6f}'.format(anomaly_score[i].item()),
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       thickness,
                       lineType)

        inputs = torch.div(torch.tensor(inputs_np).to(torch.float32), 255).permute(0, 3, 1, 2)

        # ['inputs', 'gt_maps', 'outputs']
        for i, main_tag in enumerate(['normal', 'anomalous']):
        #for inp, gt, out in zip(inputs, gt_maps, outputs):
            #if imgs is not None:
            img_list = []
            if gt_maps is None:
                for inp, out in zip(inputs[labels == i], outputs[labels == i]):
                    img_list.append(inp)
                    img_list.append(out)
                #img_list = [inputs[labels == i], outputs[labels == i]]
            else:
                if gt_maps is None:
                    for inp, out in zip(inputs[labels == i], outputs[labels == i]):
                        img_list.append(inp)
                        img_list.append(out)
                else:
                    for inp, out, gt in zip(inputs[labels == i], outputs[labels == i], gt_maps[labels == i]):
                        img_list.append(inp)
                        img_list.append(out)
                        img_list.append(gt)
            #batch_size = imgs2.size(0)
            # nrow = int(np.sqrt(batch_size))
            grid = make_grid(img_list, nrow=2 if gt_maps is None else 3)
            self.writer.add_image(main_tag, grid, epoch)
        self.writer.flush()

    def add_network(self, model: torch.nn.Module, input_to_model):
        self.writer.add_graph(model, input_to_model)
        self.writer.flush()

    def add_weight_histograms(self, model, epoch):
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                self.writer.add_histogram(name + '.weight', m.weight, epoch)
                if m.bias is not None:
                    self.writer.add_histogram(name + '.bias', m.bias, epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()
