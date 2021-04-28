from pathlib import Path
from typing import List

import torch
from models.experimental import attempt_load
from torch import Tensor
from torch import device as Device
from utils.general import non_max_suppression


File = Path

default_device = torch.device('cpu')


class YoloFacade(torch.nn.Module):
    '''Exposes traced NN to final inference via TorchScript.

    Loads the traced trained model and perform `forward` method of it
    with post-processing.
    '''

    def __init__(
        self,
        class_names: List[str],
        traced_path: File,
        stride: Tensor,
        anchor_grid: Tensor,
        conf_thres: float = 0.1,
        iou_thres: float = 0.6,
        agnostic_nms: bool = True,
        device: Device = default_device,
        version: str = '1.0.0'
    ):
        '''
        Args:
            class_names: list of string labels of classes
            traced_path: path to traced model
            stride, anchor_grid: params from full fledged model to convert bboxes
                to original scale
            conf_thres, iou_thres, agnostic_nms: params of NMS
            device: cuda or cpu
            version: version number string n.n.n formatted
        '''
        super().__init__()

        self.names = class_names
        self.device = device

        self.stride = stride.to(self.device)
        self.anchor_grid = anchor_grid.to(self.device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        
        self.version = version

        self.model = torch.jit.load(traced_path.as_posix(), map_location=self.device)
        self.model.eval()

        self._grid = [torch.tensor(()), torch.tensor(()), torch.tensor(())]

    def forward(self, batch: Tensor) -> List[Tensor]:
        '''
        Args:
            batch: 4D tensor (BxHxWxC) of stacked preprocessed images

        Returns:
            List (of len B) of prediction tensors for each image in batch.
            Each tensor is (#predictions_on_image, 6) shaped.
            First 4 values in prediction is tlbr coordinates of detected bbox,
                next is confidence and last is class label
                (class names available in `names` attribute)
        '''
        batch = batch.to(self.device)
        raw = self.model(batch)

        res = []
        for i in range(len(raw)):
            bs, na, ny, nx, no = raw[i].shape
            if self._grid[i].shape[2:4] != raw[i].shape[2:4]:
                self._grid[i] = self._make_grid(nx, ny)

            y = raw[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self._grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            res.append(y.view(bs, -1, no))

        return non_max_suppression(
            torch.cat(res, 1), self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms, scripting=True
        )

    def _make_grid(self, nx: int = 20, ny: int = 20):
        yv, xv = torch.meshgrid(
            (torch.arange(ny, device=self.device), torch.arange(nx, device=self.device))
        )
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @classmethod
    def from_checkpoint(
        cls, 
        yolo_path: File, 
        traced_path: File, 
        device: Device, 
        iou_thres: float, 
        conf_thres: float,
        version: str
    ):
        '''Loads class from checkpoint(s)

        Args:
            see `script`
        '''
        model = attempt_load(yolo_path.as_posix())
        detect = model.model[-1]
        stride = detect.stride.clone().detach()
        anchor_grid = detect.anchor_grid.clone().detach()

        return cls(model.names, 
                   traced_path, 
                   stride, 
                   anchor_grid, 
                   device=device, 
                   iou_thres=iou_thres, 
                   conf_thres=conf_thres,
                   version=version)

    @classmethod
    def script(
        cls, 
        yolo_path: File, 
        traced_path: File, 
        scripted_path: File, 
        device: Device, 
        iou_thres: float = 0.6, 
        conf_thres: float = 0.1,
        version: str = '1.0.0'
    ):
        '''Saves scripted version of Facade to scripted_path

        Args:
            yolo_path: original model dump (needed to get parameters which is not traced)
            traced_path: tha path to traced trained model
            scripted_path: the path for scripted model
            device: the device
            
            iou_thres, conf_thres, version: see `__init__`
        '''
        facade = cls.from_checkpoint(yolo_path, traced_path, device, iou_thres, conf_thres, version)
        scripted = torch.jit.script(facade)
        scripted.save(scripted_path.as_posix())

    @classmethod
    def load(cls, scripted_path: File, device: Device):
        '''Loads previously scripted Facade.
        '''
        return torch.jit.load(scripted_path.as_posix(), device)
