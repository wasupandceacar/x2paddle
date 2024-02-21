import paddle
import math
from x2paddle.op_mapper.pytorch2paddle import pytorch_custom_layer as x2paddle_nn

class U2NETF_LBP(paddle.nn.Layer):
    def __init__(self):
        super(U2NETF_LBP, self).__init__()
        self.conv2d0 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=4)
        self.batchnorm0 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.conv2d1 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=32)
        self.batchnorm1 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.pool2d0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d2 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm2 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
        self.pool2d1 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d3 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm3 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu3 = paddle.nn.ReLU()
        self.pool2d2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d4 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm4 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu4 = paddle.nn.ReLU()
        self.pool2d3 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d5 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm5 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu5 = paddle.nn.ReLU()
        self.pool2d4 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d6 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm6 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu6 = paddle.nn.ReLU()
        self.conv2d7 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm7 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu7 = paddle.nn.ReLU()
        self.conv2d8 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm8 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu8 = paddle.nn.ReLU()
        self.conv2d9 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm9 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu9 = paddle.nn.ReLU()
        self.conv2d10 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm10 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu10 = paddle.nn.ReLU()
        self.conv2d11 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm11 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu11 = paddle.nn.ReLU()
        self.conv2d12 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm12 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu12 = paddle.nn.ReLU()
        self.conv2d13 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=16)
        self.batchnorm13 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu13 = paddle.nn.ReLU()
        self.pool2d5 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d14 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=32)
        self.batchnorm14 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu14 = paddle.nn.ReLU()
        self.conv2d15 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=32)
        self.batchnorm15 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu15 = paddle.nn.ReLU()
        self.pool2d6 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d16 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm16 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu16 = paddle.nn.ReLU()
        self.pool2d7 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d17 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm17 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu17 = paddle.nn.ReLU()
        self.pool2d8 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d18 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm18 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu18 = paddle.nn.ReLU()
        self.pool2d9 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d19 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm19 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu19 = paddle.nn.ReLU()
        self.conv2d20 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm20 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu20 = paddle.nn.ReLU()
        self.conv2d21 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm21 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu21 = paddle.nn.ReLU()
        self.conv2d22 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm22 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu22 = paddle.nn.ReLU()
        self.conv2d23 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm23 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu23 = paddle.nn.ReLU()
        self.conv2d24 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm24 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu24 = paddle.nn.ReLU()
        self.conv2d25 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=16)
        self.batchnorm25 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu25 = paddle.nn.ReLU()
        self.pool2d10 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d26 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm26 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu26 = paddle.nn.ReLU()
        self.conv2d27 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=64)
        self.batchnorm27 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu27 = paddle.nn.ReLU()
        self.pool2d11 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d28 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm28 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu28 = paddle.nn.ReLU()
        self.pool2d12 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d29 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm29 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu29 = paddle.nn.ReLU()
        self.pool2d13 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d30 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm30 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu30 = paddle.nn.ReLU()
        self.conv2d31 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm31 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu31 = paddle.nn.ReLU()
        self.conv2d32 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm32 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu32 = paddle.nn.ReLU()
        self.conv2d33 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm33 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu33 = paddle.nn.ReLU()
        self.conv2d34 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm34 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu34 = paddle.nn.ReLU()
        self.conv2d35 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=16)
        self.batchnorm35 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu35 = paddle.nn.ReLU()
        self.pool2d14 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d36 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.batchnorm36 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu36 = paddle.nn.ReLU()
        self.conv2d37 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm37 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu37 = paddle.nn.ReLU()
        self.pool2d15 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d38 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm38 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu38 = paddle.nn.ReLU()
        self.pool2d16 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d39 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm39 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu39 = paddle.nn.ReLU()
        self.conv2d40 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm40 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu40 = paddle.nn.ReLU()
        self.conv2d41 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm41 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu41 = paddle.nn.ReLU()
        self.conv2d42 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm42 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu42 = paddle.nn.ReLU()
        self.conv2d43 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm43 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu43 = paddle.nn.ReLU()
        self.pool2d17 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d44 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.batchnorm44 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu44 = paddle.nn.ReLU()
        self.conv2d45 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm45 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu45 = paddle.nn.ReLU()
        self.conv2d46 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm46 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu46 = paddle.nn.ReLU()
        self.conv2d47 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm47 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu47 = paddle.nn.ReLU()
        self.conv2d48 = paddle.nn.Conv2D(padding=8, dilation=8, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm48 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu48 = paddle.nn.ReLU()
        self.conv2d49 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm49 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu49 = paddle.nn.ReLU()
        self.conv2d50 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm50 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu50 = paddle.nn.ReLU()
        self.conv2d51 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm51 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu51 = paddle.nn.ReLU()
        self.pool2d18 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d52 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=64)
        self.batchnorm52 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu52 = paddle.nn.ReLU()
        self.conv2d53 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm53 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu53 = paddle.nn.ReLU()
        self.conv2d54 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm54 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu54 = paddle.nn.ReLU()
        self.conv2d55 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm55 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu55 = paddle.nn.ReLU()
        self.conv2d56 = paddle.nn.Conv2D(padding=8, dilation=8, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm56 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu56 = paddle.nn.ReLU()
        self.conv2d57 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm57 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu57 = paddle.nn.ReLU()
        self.conv2d58 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm58 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu58 = paddle.nn.ReLU()
        self.conv2d59 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm59 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu59 = paddle.nn.ReLU()
        self.conv2d60 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=128)
        self.batchnorm60 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu60 = paddle.nn.ReLU()
        self.conv2d61 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm61 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu61 = paddle.nn.ReLU()
        self.conv2d62 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm62 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu62 = paddle.nn.ReLU()
        self.conv2d63 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm63 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu63 = paddle.nn.ReLU()
        self.conv2d64 = paddle.nn.Conv2D(padding=8, dilation=8, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm64 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu64 = paddle.nn.ReLU()
        self.conv2d65 = paddle.nn.Conv2D(padding=4, dilation=4, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm65 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu65 = paddle.nn.ReLU()
        self.conv2d66 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm66 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu66 = paddle.nn.ReLU()
        self.conv2d67 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm67 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu67 = paddle.nn.ReLU()
        self.conv2d68 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=128)
        self.batchnorm68 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu68 = paddle.nn.ReLU()
        self.conv2d69 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm69 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu69 = paddle.nn.ReLU()
        self.pool2d19 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d70 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm70 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu70 = paddle.nn.ReLU()
        self.pool2d20 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d71 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm71 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu71 = paddle.nn.ReLU()
        self.conv2d72 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm72 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu72 = paddle.nn.ReLU()
        self.conv2d73 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm73 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu73 = paddle.nn.ReLU()
        self.conv2d74 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm74 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu74 = paddle.nn.ReLU()
        self.conv2d75 = paddle.nn.Conv2D(padding=1, out_channels=64, kernel_size=(3, 3), in_channels=32)
        self.batchnorm75 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu75 = paddle.nn.ReLU()
        self.conv2d76 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=128)
        self.batchnorm76 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu76 = paddle.nn.ReLU()
        self.conv2d77 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm77 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu77 = paddle.nn.ReLU()
        self.pool2d21 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d78 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm78 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu78 = paddle.nn.ReLU()
        self.pool2d22 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d79 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm79 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu79 = paddle.nn.ReLU()
        self.pool2d23 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d80 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm80 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu80 = paddle.nn.ReLU()
        self.conv2d81 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm81 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu81 = paddle.nn.ReLU()
        self.conv2d82 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm82 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu82 = paddle.nn.ReLU()
        self.conv2d83 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm83 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu83 = paddle.nn.ReLU()
        self.conv2d84 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=32)
        self.batchnorm84 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu84 = paddle.nn.ReLU()
        self.conv2d85 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=32)
        self.batchnorm85 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu85 = paddle.nn.ReLU()
        self.conv2d86 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=64)
        self.batchnorm86 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu86 = paddle.nn.ReLU()
        self.conv2d87 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=32)
        self.batchnorm87 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu87 = paddle.nn.ReLU()
        self.pool2d24 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d88 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm88 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu88 = paddle.nn.ReLU()
        self.pool2d25 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d89 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm89 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu89 = paddle.nn.ReLU()
        self.pool2d26 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d90 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm90 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu90 = paddle.nn.ReLU()
        self.pool2d27 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d91 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm91 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu91 = paddle.nn.ReLU()
        self.conv2d92 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm92 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu92 = paddle.nn.ReLU()
        self.conv2d93 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm93 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu93 = paddle.nn.ReLU()
        self.conv2d94 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm94 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu94 = paddle.nn.ReLU()
        self.conv2d95 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm95 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu95 = paddle.nn.ReLU()
        self.conv2d96 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm96 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu96 = paddle.nn.ReLU()
        self.conv2d97 = paddle.nn.Conv2D(padding=1, out_channels=32, kernel_size=(3, 3), in_channels=16)
        self.batchnorm97 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1)
        self.relu97 = paddle.nn.ReLU()
        self.conv2d98 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=64)
        self.batchnorm98 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu98 = paddle.nn.ReLU()
        self.conv2d99 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm99 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu99 = paddle.nn.ReLU()
        self.pool2d28 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d100 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm100 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu100 = paddle.nn.ReLU()
        self.pool2d29 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d101 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm101 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu101 = paddle.nn.ReLU()
        self.pool2d30 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d102 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm102 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu102 = paddle.nn.ReLU()
        self.pool2d31 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d103 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm103 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu103 = paddle.nn.ReLU()
        self.pool2d32 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv2d104 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm104 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu104 = paddle.nn.ReLU()
        self.conv2d105 = paddle.nn.Conv2D(padding=2, dilation=2, out_channels=8, kernel_size=(3, 3), in_channels=8)
        self.batchnorm105 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu105 = paddle.nn.ReLU()
        self.conv2d106 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm106 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu106 = paddle.nn.ReLU()
        self.conv2d107 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm107 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu107 = paddle.nn.ReLU()
        self.conv2d108 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm108 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu108 = paddle.nn.ReLU()
        self.conv2d109 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm109 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu109 = paddle.nn.ReLU()
        self.conv2d110 = paddle.nn.Conv2D(padding=1, out_channels=8, kernel_size=(3, 3), in_channels=16)
        self.batchnorm110 = paddle.nn.BatchNorm(is_test=True, num_channels=8, momentum=0.1)
        self.relu110 = paddle.nn.ReLU()
        self.conv2d111 = paddle.nn.Conv2D(padding=1, out_channels=16, kernel_size=(3, 3), in_channels=16)
        self.batchnorm111 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1)
        self.relu111 = paddle.nn.ReLU()
        self.conv2d112 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=16)
        self.conv2d113 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=32)
        self.conv2d114 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=32)
        self.conv2d115 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=64)
        self.conv2d116 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=64)
        self.conv2d117 = paddle.nn.Conv2D(padding=1, out_channels=2, kernel_size=(3, 3), in_channels=64)
        self.conv2d118 = paddle.nn.Conv2D(out_channels=2, kernel_size=(1, 1), in_channels=12)

    def forward(self, x0):
        x19_list = [0]
        x20_list = [0]
        x21_list = [2147483647]
        x22_list = [1]
        x23 = paddle.strided_slice(x=x0, axes=x19_list, starts=x20_list, ends=x21_list, strides=x22_list)
        x25 = 0
        x26 = x23[:, x25]
        x27_list = [1]
        x28_list = [0]
        x29_list = [2147483647]
        x30_list = [1]
        x31 = paddle.strided_slice(x=x26, axes=x27_list, starts=x28_list, ends=x29_list, strides=x30_list)
        x32_list = [2]
        x33_list = [0]
        x34_list = [2147483647]
        x35_list = [1]
        x36 = paddle.strided_slice(x=x31, axes=x32_list, starts=x33_list, ends=x34_list, strides=x35_list)
        x37 = 0.299
        x38 = x36 * x37
        x39_list = [0]
        x40_list = [0]
        x41_list = [2147483647]
        x42_list = [1]
        x43 = paddle.strided_slice(x=x0, axes=x39_list, starts=x40_list, ends=x41_list, strides=x42_list)
        x45 = 1
        x46 = x43[:, x45]
        x47_list = [1]
        x48_list = [0]
        x49_list = [2147483647]
        x50_list = [1]
        x51 = paddle.strided_slice(x=x46, axes=x47_list, starts=x48_list, ends=x49_list, strides=x50_list)
        x52_list = [2]
        x53_list = [0]
        x54_list = [2147483647]
        x55_list = [1]
        x56 = paddle.strided_slice(x=x51, axes=x52_list, starts=x53_list, ends=x54_list, strides=x55_list)
        x57 = 0.587
        x58 = x56 * x57
        x60 = x38 + x58
        x61_list = [0]
        x62_list = [0]
        x63_list = [2147483647]
        x64_list = [1]
        x65 = paddle.strided_slice(x=x0, axes=x61_list, starts=x62_list, ends=x63_list, strides=x64_list)
        x67 = 2
        x68 = x65[:, x67]
        x69_list = [1]
        x70_list = [0]
        x71_list = [2147483647]
        x72_list = [1]
        x73 = paddle.strided_slice(x=x68, axes=x69_list, starts=x70_list, ends=x71_list, strides=x72_list)
        x74_list = [2]
        x75_list = [0]
        x76_list = [2147483647]
        x77_list = [1]
        x78 = paddle.strided_slice(x=x73, axes=x74_list, starts=x75_list, ends=x76_list, strides=x77_list)
        x79 = 0.144
        x80 = x78 * x79
        x82 = x60 + x80
        x84 = paddle.unsqueeze(x=x82, axis=1)
        x85 = [1, 1, 1, 1]
        x86 = 'constant'
        x88 = paddle.nn.functional.pad(x=x84, pad=x85, mode=x86)
        x93_list = [2]
        x94_list = [0]
        x91_list = [288]
        x95_list = [1]
        x96 = paddle.strided_slice(x=x88, axes=x93_list, starts=x94_list, ends=x91_list, strides=x95_list)
        x97_list = [3]
        x98_list = [0]
        x92_list = [288]
        x99_list = [1]
        x100 = paddle.strided_slice(x=x96, axes=x97_list, starts=x98_list, ends=x92_list, strides=x99_list)
        x103_list = [2]
        x104_list = [0]
        x101_list = [288]
        x105_list = [1]
        x106 = paddle.strided_slice(x=x88, axes=x103_list, starts=x104_list, ends=x101_list, strides=x105_list)
        x107_list = [3]
        x108_list = [1]
        x102_list = [289]
        x109_list = [1]
        x110 = paddle.strided_slice(x=x106, axes=x107_list, starts=x108_list, ends=x102_list, strides=x109_list)
        x112_list = [2]
        x113_list = [0]
        x111_list = [288]
        x114_list = [1]
        x115 = paddle.strided_slice(x=x88, axes=x112_list, starts=x113_list, ends=x111_list, strides=x114_list)
        x116_list = [3]
        x117_list = [2]
        x90_list = [290]
        x118_list = [1]
        x119 = paddle.strided_slice(x=x115, axes=x116_list, starts=x117_list, ends=x90_list, strides=x118_list)
        x122_list = [2]
        x123_list = [1]
        x120_list = [289]
        x124_list = [1]
        x125 = paddle.strided_slice(x=x88, axes=x122_list, starts=x123_list, ends=x120_list, strides=x124_list)
        x126_list = [3]
        x127_list = [0]
        x121_list = [288]
        x128_list = [1]
        x129 = paddle.strided_slice(x=x125, axes=x126_list, starts=x127_list, ends=x121_list, strides=x128_list)
        x132_list = [2]
        x133_list = [1]
        x130_list = [289]
        x134_list = [1]
        x135 = paddle.strided_slice(x=x88, axes=x132_list, starts=x133_list, ends=x130_list, strides=x134_list)
        x136_list = [3]
        x137_list = [1]
        x131_list = [289]
        x138_list = [1]
        x139 = paddle.strided_slice(x=x135, axes=x136_list, starts=x137_list, ends=x131_list, strides=x138_list)
        x141_list = [2]
        x142_list = [1]
        x140_list = [289]
        x143_list = [1]
        x144 = paddle.strided_slice(x=x88, axes=x141_list, starts=x142_list, ends=x140_list, strides=x143_list)
        x145_list = [3]
        x146_list = [2]
        x90_list = [290]
        x147_list = [1]
        x148 = paddle.strided_slice(x=x144, axes=x145_list, starts=x146_list, ends=x90_list, strides=x147_list)
        x150_list = [2]
        x151_list = [2]
        x89_list = [290]
        x152_list = [1]
        x153 = paddle.strided_slice(x=x88, axes=x150_list, starts=x151_list, ends=x89_list, strides=x152_list)
        x154_list = [3]
        x155_list = [0]
        x149_list = [288]
        x156_list = [1]
        x157 = paddle.strided_slice(x=x153, axes=x154_list, starts=x155_list, ends=x149_list, strides=x156_list)
        x159_list = [2]
        x160_list = [2]
        x89_list = [290]
        x161_list = [1]
        x162 = paddle.strided_slice(x=x88, axes=x159_list, starts=x160_list, ends=x89_list, strides=x161_list)
        x163_list = [3]
        x164_list = [1]
        x158_list = [289]
        x165_list = [1]
        x166 = paddle.strided_slice(x=x162, axes=x163_list, starts=x164_list, ends=x158_list, strides=x165_list)
        x167_list = [2]
        x168_list = [2]
        x89_list = [290]
        x169_list = [1]
        x170 = paddle.strided_slice(x=x88, axes=x167_list, starts=x168_list, ends=x89_list, strides=x169_list)
        x171_list = [3]
        x172_list = [2]
        x90_list = [290]
        x173_list = [1]
        x174 = paddle.strided_slice(x=x170, axes=x171_list, starts=x172_list, ends=x90_list, strides=x173_list)
        x175 = x110 >= x139
        x176 = 1
        x177 = x175 * x176
        x178 = x119 >= x139
        x179 = 2
        x180 = x178 * x179
        x182 = x180 + x177
        x183 = x148 >= x139
        x184 = 4
        x185 = x183 * x184
        x187 = x182 + x185
        x188 = x174 >= x139
        x189 = 8
        x190 = x188 * x189
        x192 = x187 + x190
        x193 = x166 >= x139
        x194 = 16
        x195 = x193 * x194
        x197 = x192 + x195
        x198 = x157 >= x139
        x199 = 32
        x200 = x198 * x199
        x202 = x197 + x200
        x203 = x129 >= x139
        x204 = 64
        x205 = x203 * x204
        x207 = x202 + x205
        x208 = x100 >= x139
        x209 = 128
        x210 = x208 * x209
        x212 = x207 + x210
        x213 = [x0, x212]
        print(x0.type, x212.type)
        x215 = paddle.concat(x=x213, axis=1)
        x244 = self.conv2d0(x215)
        x249 = self.batchnorm0(x244)
        x250 = self.relu0(x249)
        x259 = self.conv2d1(x250)
        x264 = self.batchnorm1(x259)
        x265 = self.relu1(x264)
        x270 = self.pool2d0(x265)
        x279 = self.conv2d2(x270)
        x284 = self.batchnorm2(x279)
        x285 = self.relu2(x284)
        x290 = self.pool2d1(x285)
        x299 = self.conv2d3(x290)
        x304 = self.batchnorm3(x299)
        x305 = self.relu3(x304)
        x310 = self.pool2d2(x305)
        x319 = self.conv2d4(x310)
        x324 = self.batchnorm4(x319)
        x325 = self.relu4(x324)
        x330 = self.pool2d3(x325)
        x339 = self.conv2d5(x330)
        x344 = self.batchnorm5(x339)
        x345 = self.relu5(x344)
        x350 = self.pool2d4(x345)
        x359 = self.conv2d6(x350)
        x364 = self.batchnorm6(x359)
        x365 = self.relu6(x364)
        x374 = self.conv2d7(x365)
        x379 = self.batchnorm7(x374)
        x380 = self.relu7(x379)
        x381 = [x380, x365]
        print(x380.type, x365.type)
        x382 = paddle.concat(x=x381, axis=1)
        x391 = self.conv2d8(x382)
        x396 = self.batchnorm8(x391)
        x397 = self.relu8(x396)
        x399 = paddle.nn.functional.interpolate(x=x397, size=[18, 18], mode='bilinear')
        x400 = [x399, x345]
        print(x399.type, x345.type)
        x401 = paddle.concat(x=x400, axis=1)
        x410 = self.conv2d9(x401)
        x415 = self.batchnorm9(x410)
        x416 = self.relu9(x415)
        x418 = paddle.nn.functional.interpolate(x=x416, size=[36, 36], mode='bilinear')
        x419 = [x418, x325]
        print(x418.type, x325.type)
        x420 = paddle.concat(x=x419, axis=1)
        x429 = self.conv2d10(x420)
        x434 = self.batchnorm10(x429)
        x435 = self.relu10(x434)
        x437 = paddle.nn.functional.interpolate(x=x435, size=[72, 72], mode='bilinear')
        x438 = [x437, x305]
        print(x437.type, x305.type)
        x439 = paddle.concat(x=x438, axis=1)
        x448 = self.conv2d11(x439)
        x453 = self.batchnorm11(x448)
        x454 = self.relu11(x453)
        x456 = paddle.nn.functional.interpolate(x=x454, size=[144, 144], mode='bilinear')
        x457 = [x456, x285]
        print(x456.type, x285.type)
        x458 = paddle.concat(x=x457, axis=1)
        x467 = self.conv2d12(x458)
        x472 = self.batchnorm12(x467)
        x473 = self.relu12(x472)
        x475 = paddle.nn.functional.interpolate(x=x473, size=[288, 288], mode='bilinear')
        x476 = [x475, x265]
        print(x475.type, x265.type)
        x477 = paddle.concat(x=x476, axis=1)
        x486 = self.conv2d13(x477)
        x491 = self.batchnorm13(x486)
        x492 = self.relu13(x491)
        x493 = x492 + x250
        x499 = self.pool2d5(x493)
        x526 = self.conv2d14(x499)
        x531 = self.batchnorm14(x526)
        x532 = self.relu14(x531)
        x541 = self.conv2d15(x532)
        x546 = self.batchnorm15(x541)
        x547 = self.relu15(x546)
        x552 = self.pool2d6(x547)
        x561 = self.conv2d16(x552)
        x566 = self.batchnorm16(x561)
        x567 = self.relu16(x566)
        x572 = self.pool2d7(x567)
        x581 = self.conv2d17(x572)
        x586 = self.batchnorm17(x581)
        x587 = self.relu17(x586)
        x592 = self.pool2d8(x587)
        x601 = self.conv2d18(x592)
        x606 = self.batchnorm18(x601)
        x607 = self.relu18(x606)
        x612 = self.pool2d9(x607)
        x621 = self.conv2d19(x612)
        x626 = self.batchnorm19(x621)
        x627 = self.relu19(x626)
        x636 = self.conv2d20(x627)
        x641 = self.batchnorm20(x636)
        x642 = self.relu20(x641)
        x643 = [x642, x627]
        print(x642.type, x627.type)
        x644 = paddle.concat(x=x643, axis=1)
        x653 = self.conv2d21(x644)
        x658 = self.batchnorm21(x653)
        x659 = self.relu21(x658)
        x661 = paddle.nn.functional.interpolate(x=x659, size=[18, 18], mode='bilinear')
        x662 = [x661, x607]
        print(x661.type, x607.type)
        x663 = paddle.concat(x=x662, axis=1)
        x672 = self.conv2d22(x663)
        x677 = self.batchnorm22(x672)
        x678 = self.relu22(x677)
        x680 = paddle.nn.functional.interpolate(x=x678, size=[36, 36], mode='bilinear')
        x681 = [x680, x587]
        print(x680.type, x587.type)
        x682 = paddle.concat(x=x681, axis=1)
        x691 = self.conv2d23(x682)
        x696 = self.batchnorm23(x691)
        x697 = self.relu23(x696)
        x699 = paddle.nn.functional.interpolate(x=x697, size=[72, 72], mode='bilinear')
        x700 = [x699, x567]
        print(x699.type, x567.type)
        x701 = paddle.concat(x=x700, axis=1)
        x710 = self.conv2d24(x701)
        x715 = self.batchnorm24(x710)
        x716 = self.relu24(x715)
        x718 = paddle.nn.functional.interpolate(x=x716, size=[144, 144], mode='bilinear')
        x719 = [x718, x547]
        print(x718.type, x547.type)
        x720 = paddle.concat(x=x719, axis=1)
        x729 = self.conv2d25(x720)
        x734 = self.batchnorm25(x729)
        x735 = self.relu25(x734)
        x736 = x735 + x532
        x742 = self.pool2d10(x736)
        x767 = self.conv2d26(x742)
        x772 = self.batchnorm26(x767)
        x773 = self.relu26(x772)
        x782 = self.conv2d27(x773)
        x787 = self.batchnorm27(x782)
        x788 = self.relu27(x787)
        x793 = self.pool2d11(x788)
        x802 = self.conv2d28(x793)
        x807 = self.batchnorm28(x802)
        x808 = self.relu28(x807)
        x813 = self.pool2d12(x808)
        x822 = self.conv2d29(x813)
        x827 = self.batchnorm29(x822)
        x828 = self.relu29(x827)
        x833 = self.pool2d13(x828)
        x842 = self.conv2d30(x833)
        x847 = self.batchnorm30(x842)
        x848 = self.relu30(x847)
        x857 = self.conv2d31(x848)
        x862 = self.batchnorm31(x857)
        x863 = self.relu31(x862)
        x864 = [x863, x848]
        print(x863.type, x848.type)
        x865 = paddle.concat(x=x864, axis=1)
        x874 = self.conv2d32(x865)
        x879 = self.batchnorm32(x874)
        x880 = self.relu32(x879)
        x882 = paddle.nn.functional.interpolate(x=x880, size=[18, 18], mode='bilinear')
        x883 = [x882, x828]
        print(x882.type, x828.type)
        x884 = paddle.concat(x=x883, axis=1)
        x893 = self.conv2d33(x884)
        x898 = self.batchnorm33(x893)
        x899 = self.relu33(x898)
        x901 = paddle.nn.functional.interpolate(x=x899, size=[36, 36], mode='bilinear')
        x902 = [x901, x808]
        print(x901.type, x808.type)
        x903 = paddle.concat(x=x902, axis=1)
        x912 = self.conv2d34(x903)
        x917 = self.batchnorm34(x912)
        x918 = self.relu34(x917)
        x920 = paddle.nn.functional.interpolate(x=x918, size=[72, 72], mode='bilinear')
        x921 = [x920, x788]
        print(x920.type, x788.type)
        x922 = paddle.concat(x=x921, axis=1)
        x931 = self.conv2d35(x922)
        x936 = self.batchnorm35(x931)
        x937 = self.relu35(x936)
        x938 = x937 + x773
        x944 = self.pool2d14(x938)
        x967 = self.conv2d36(x944)
        x972 = self.batchnorm36(x967)
        x973 = self.relu36(x972)
        x982 = self.conv2d37(x973)
        x987 = self.batchnorm37(x982)
        x988 = self.relu37(x987)
        x993 = self.pool2d15(x988)
        x1002 = self.conv2d38(x993)
        x1007 = self.batchnorm38(x1002)
        x1008 = self.relu38(x1007)
        x1013 = self.pool2d16(x1008)
        x1022 = self.conv2d39(x1013)
        x1027 = self.batchnorm39(x1022)
        x1028 = self.relu39(x1027)
        x1037 = self.conv2d40(x1028)
        x1042 = self.batchnorm40(x1037)
        x1043 = self.relu40(x1042)
        x1044 = [x1043, x1028]
        print(x1043.type, x1028.type)
        x1045 = paddle.concat(x=x1044, axis=1)
        x1054 = self.conv2d41(x1045)
        x1059 = self.batchnorm41(x1054)
        x1060 = self.relu41(x1059)
        x1062 = paddle.nn.functional.interpolate(x=x1060, size=[18, 18], mode='bilinear')
        x1063 = [x1062, x1008]
        print(x1062.type, x1008.type)
        x1064 = paddle.concat(x=x1063, axis=1)
        x1073 = self.conv2d42(x1064)
        x1078 = self.batchnorm42(x1073)
        x1079 = self.relu42(x1078)
        x1081 = paddle.nn.functional.interpolate(x=x1079, size=[36, 36], mode='bilinear')
        x1082 = [x1081, x988]
        print(x1081.type, x988.type)
        x1083 = paddle.concat(x=x1082, axis=1)
        x1092 = self.conv2d43(x1083)
        x1097 = self.batchnorm43(x1092)
        x1098 = self.relu43(x1097)
        x1099 = x1098 + x973
        x1105 = self.pool2d17(x1099)
        x1127 = self.conv2d44(x1105)
        x1132 = self.batchnorm44(x1127)
        x1133 = self.relu44(x1132)
        x1142 = self.conv2d45(x1133)
        x1147 = self.batchnorm45(x1142)
        x1148 = self.relu45(x1147)
        x1157 = self.conv2d46(x1148)
        x1162 = self.batchnorm46(x1157)
        x1163 = self.relu46(x1162)
        x1172 = self.conv2d47(x1163)
        x1177 = self.batchnorm47(x1172)
        x1178 = self.relu47(x1177)
        x1187 = self.conv2d48(x1178)
        x1192 = self.batchnorm48(x1187)
        x1193 = self.relu48(x1192)
        x1194 = [x1193, x1178]
        print(x1193.type, x1178.type)
        x1195 = paddle.concat(x=x1194, axis=1)
        x1204 = self.conv2d49(x1195)
        x1209 = self.batchnorm49(x1204)
        x1210 = self.relu49(x1209)
        x1211 = [x1210, x1163]
        print(x1210.type, x1163.type)
        x1212 = paddle.concat(x=x1211, axis=1)
        x1221 = self.conv2d50(x1212)
        x1226 = self.batchnorm50(x1221)
        x1227 = self.relu50(x1226)
        x1228 = [x1227, x1148]
        print(x1227.type, x1148.type)
        x1229 = paddle.concat(x=x1228, axis=1)
        x1238 = self.conv2d51(x1229)
        x1243 = self.batchnorm51(x1238)
        x1244 = self.relu51(x1243)
        x1245 = x1244 + x1133
        x1251 = self.pool2d18(x1245)
        x1273 = self.conv2d52(x1251)
        x1278 = self.batchnorm52(x1273)
        x1279 = self.relu52(x1278)
        x1288 = self.conv2d53(x1279)
        x1293 = self.batchnorm53(x1288)
        x1294 = self.relu53(x1293)
        x1303 = self.conv2d54(x1294)
        x1308 = self.batchnorm54(x1303)
        x1309 = self.relu54(x1308)
        x1318 = self.conv2d55(x1309)
        x1323 = self.batchnorm55(x1318)
        x1324 = self.relu55(x1323)
        x1333 = self.conv2d56(x1324)
        x1338 = self.batchnorm56(x1333)
        x1339 = self.relu56(x1338)
        x1340 = [x1339, x1324]
        print(x1339.type, x1324.type)
        x1341 = paddle.concat(x=x1340, axis=1)
        x1350 = self.conv2d57(x1341)
        x1355 = self.batchnorm57(x1350)
        x1356 = self.relu57(x1355)
        x1357 = [x1356, x1309]
        print(x1356.type, x1309.type)
        x1358 = paddle.concat(x=x1357, axis=1)
        x1367 = self.conv2d58(x1358)
        x1372 = self.batchnorm58(x1367)
        x1373 = self.relu58(x1372)
        x1374 = [x1373, x1294]
        print(x1373.type, x1294.type)
        x1375 = paddle.concat(x=x1374, axis=1)
        x1384 = self.conv2d59(x1375)
        x1389 = self.batchnorm59(x1384)
        x1390 = self.relu59(x1389)
        x1391 = x1390 + x1279
        x1395 = paddle.nn.functional.interpolate(x=x1391, size=[18, 18], mode='bilinear')
        x1396 = [x1395, x1245]
        print(x1395.type, x1245.type)
        x1398 = paddle.concat(x=x1396, axis=1)
        x1420 = self.conv2d60(x1398)
        x1425 = self.batchnorm60(x1420)
        x1426 = self.relu60(x1425)
        x1435 = self.conv2d61(x1426)
        x1440 = self.batchnorm61(x1435)
        x1441 = self.relu61(x1440)
        x1450 = self.conv2d62(x1441)
        x1455 = self.batchnorm62(x1450)
        x1456 = self.relu62(x1455)
        x1465 = self.conv2d63(x1456)
        x1470 = self.batchnorm63(x1465)
        x1471 = self.relu63(x1470)
        x1480 = self.conv2d64(x1471)
        x1485 = self.batchnorm64(x1480)
        x1486 = self.relu64(x1485)
        x1487 = [x1486, x1471]
        print(x1486.type, x1471.type)
        x1488 = paddle.concat(x=x1487, axis=1)
        x1497 = self.conv2d65(x1488)
        x1502 = self.batchnorm65(x1497)
        x1503 = self.relu65(x1502)
        x1504 = [x1503, x1456]
        print(x1503.type, x1456.type)
        x1505 = paddle.concat(x=x1504, axis=1)
        x1514 = self.conv2d66(x1505)
        x1519 = self.batchnorm66(x1514)
        x1520 = self.relu66(x1519)
        x1521 = [x1520, x1441]
        print(x1520.type, x1441.type)
        x1522 = paddle.concat(x=x1521, axis=1)
        x1531 = self.conv2d67(x1522)
        x1536 = self.batchnorm67(x1531)
        x1537 = self.relu67(x1536)
        x1538 = x1537 + x1426
        x1542 = paddle.nn.functional.interpolate(x=x1538, size=[36, 36], mode='bilinear')
        x1543 = [x1542, x1099]
        print(x1542.type, x1099.type)
        x1545 = paddle.concat(x=x1543, axis=1)
        x1568 = self.conv2d68(x1545)
        x1573 = self.batchnorm68(x1568)
        x1574 = self.relu68(x1573)
        x1583 = self.conv2d69(x1574)
        x1588 = self.batchnorm69(x1583)
        x1589 = self.relu69(x1588)
        x1594 = self.pool2d19(x1589)
        x1603 = self.conv2d70(x1594)
        x1608 = self.batchnorm70(x1603)
        x1609 = self.relu70(x1608)
        x1614 = self.pool2d20(x1609)
        x1623 = self.conv2d71(x1614)
        x1628 = self.batchnorm71(x1623)
        x1629 = self.relu71(x1628)
        x1638 = self.conv2d72(x1629)
        x1643 = self.batchnorm72(x1638)
        x1644 = self.relu72(x1643)
        x1645 = [x1644, x1629]
        print(x1644.type, x1629.type)
        x1646 = paddle.concat(x=x1645, axis=1)
        x1655 = self.conv2d73(x1646)
        x1660 = self.batchnorm73(x1655)
        x1661 = self.relu73(x1660)
        x1663 = paddle.nn.functional.interpolate(x=x1661, size=[18, 18], mode='bilinear')
        x1664 = [x1663, x1609]
        print(x1663.type, x1609.type)
        x1665 = paddle.concat(x=x1664, axis=1)
        x1674 = self.conv2d74(x1665)
        x1679 = self.batchnorm74(x1674)
        x1680 = self.relu74(x1679)
        x1682 = paddle.nn.functional.interpolate(x=x1680, size=[36, 36], mode='bilinear')
        x1683 = [x1682, x1589]
        print(x1682.type, x1589.type)
        x1684 = paddle.concat(x=x1683, axis=1)
        x1693 = self.conv2d75(x1684)
        x1698 = self.batchnorm75(x1693)
        x1699 = self.relu75(x1698)
        x1700 = x1699 + x1574
        x1704 = paddle.nn.functional.interpolate(x=x1700, size=[72, 72], mode='bilinear')
        x1705 = [x1704, x938]
        print(x1704.type, x938.type)
        x1707 = paddle.concat(x=x1705, axis=1)
        x1732 = self.conv2d76(x1707)
        x1737 = self.batchnorm76(x1732)
        x1738 = self.relu76(x1737)
        x1747 = self.conv2d77(x1738)
        x1752 = self.batchnorm77(x1747)
        x1753 = self.relu77(x1752)
        x1758 = self.pool2d21(x1753)
        x1767 = self.conv2d78(x1758)
        x1772 = self.batchnorm78(x1767)
        x1773 = self.relu78(x1772)
        x1778 = self.pool2d22(x1773)
        x1787 = self.conv2d79(x1778)
        x1792 = self.batchnorm79(x1787)
        x1793 = self.relu79(x1792)
        x1798 = self.pool2d23(x1793)
        x1807 = self.conv2d80(x1798)
        x1812 = self.batchnorm80(x1807)
        x1813 = self.relu80(x1812)
        x1822 = self.conv2d81(x1813)
        x1827 = self.batchnorm81(x1822)
        x1828 = self.relu81(x1827)
        x1829 = [x1828, x1813]
        print(x1828.type, x1813.type)
        x1830 = paddle.concat(x=x1829, axis=1)
        x1839 = self.conv2d82(x1830)
        x1844 = self.batchnorm82(x1839)
        x1845 = self.relu82(x1844)
        x1847 = paddle.nn.functional.interpolate(x=x1845, size=[18, 18], mode='bilinear')
        x1848 = [x1847, x1793]
        print(x1847.type, x1793.type)
        x1849 = paddle.concat(x=x1848, axis=1)
        x1858 = self.conv2d83(x1849)
        x1863 = self.batchnorm83(x1858)
        x1864 = self.relu83(x1863)
        x1866 = paddle.nn.functional.interpolate(x=x1864, size=[36, 36], mode='bilinear')
        x1867 = [x1866, x1773]
        print(x1866.type, x1773.type)
        x1868 = paddle.concat(x=x1867, axis=1)
        x1877 = self.conv2d84(x1868)
        x1882 = self.batchnorm84(x1877)
        x1883 = self.relu84(x1882)
        x1885 = paddle.nn.functional.interpolate(x=x1883, size=[72, 72], mode='bilinear')
        x1886 = [x1885, x1753]
        print(x1885.type, x1753.type)
        x1887 = paddle.concat(x=x1886, axis=1)
        x1896 = self.conv2d85(x1887)
        x1901 = self.batchnorm85(x1896)
        x1902 = self.relu85(x1901)
        x1903 = x1902 + x1738
        x1907 = paddle.nn.functional.interpolate(x=x1903, size=[144, 144], mode='bilinear')
        x1908 = [x1907, x736]
        print(x1907.type, x736.type)
        x1910 = paddle.concat(x=x1908, axis=1)
        x1937 = self.conv2d86(x1910)
        x1942 = self.batchnorm86(x1937)
        x1943 = self.relu86(x1942)
        x1952 = self.conv2d87(x1943)
        x1957 = self.batchnorm87(x1952)
        x1958 = self.relu87(x1957)
        x1963 = self.pool2d24(x1958)
        x1972 = self.conv2d88(x1963)
        x1977 = self.batchnorm88(x1972)
        x1978 = self.relu88(x1977)
        x1983 = self.pool2d25(x1978)
        x1992 = self.conv2d89(x1983)
        x1997 = self.batchnorm89(x1992)
        x1998 = self.relu89(x1997)
        x2003 = self.pool2d26(x1998)
        x2012 = self.conv2d90(x2003)
        x2017 = self.batchnorm90(x2012)
        x2018 = self.relu90(x2017)
        x2023 = self.pool2d27(x2018)
        x2032 = self.conv2d91(x2023)
        x2037 = self.batchnorm91(x2032)
        x2038 = self.relu91(x2037)
        x2047 = self.conv2d92(x2038)
        x2052 = self.batchnorm92(x2047)
        x2053 = self.relu92(x2052)
        x2054 = [x2053, x2038]
        print(x2053.type, x2038.type)
        x2055 = paddle.concat(x=x2054, axis=1)
        x2064 = self.conv2d93(x2055)
        x2069 = self.batchnorm93(x2064)
        x2070 = self.relu93(x2069)
        x2072 = paddle.nn.functional.interpolate(x=x2070, size=[18, 18], mode='bilinear')
        x2073 = [x2072, x2018]
        print(x2072.type, x2018.type)
        x2074 = paddle.concat(x=x2073, axis=1)
        x2083 = self.conv2d94(x2074)
        x2088 = self.batchnorm94(x2083)
        x2089 = self.relu94(x2088)
        x2091 = paddle.nn.functional.interpolate(x=x2089, size=[36, 36], mode='bilinear')
        x2092 = [x2091, x1998]
        print(x2091.type, x1998.type)
        x2093 = paddle.concat(x=x2092, axis=1)
        x2102 = self.conv2d95(x2093)
        x2107 = self.batchnorm95(x2102)
        x2108 = self.relu95(x2107)
        x2110 = paddle.nn.functional.interpolate(x=x2108, size=[72, 72], mode='bilinear')
        x2111 = [x2110, x1978]
        print(x2110.type, x1978.type)
        x2112 = paddle.concat(x=x2111, axis=1)
        x2121 = self.conv2d96(x2112)
        x2126 = self.batchnorm96(x2121)
        x2127 = self.relu96(x2126)
        x2129 = paddle.nn.functional.interpolate(x=x2127, size=[144, 144], mode='bilinear')
        x2130 = [x2129, x1958]
        print(x2129.type, x1958.type)
        x2131 = paddle.concat(x=x2130, axis=1)
        x2140 = self.conv2d97(x2131)
        x2145 = self.batchnorm97(x2140)
        x2146 = self.relu97(x2145)
        x2147 = x2146 + x1943
        x2151 = paddle.nn.functional.interpolate(x=x2147, size=[288, 288], mode='bilinear')
        x2152 = [x2151, x493]
        print(x2151.type, x493.type)
        x2154 = paddle.concat(x=x2152, axis=1)
        x2183 = self.conv2d98(x2154)
        x2188 = self.batchnorm98(x2183)
        x2189 = self.relu98(x2188)
        x2198 = self.conv2d99(x2189)
        x2203 = self.batchnorm99(x2198)
        x2204 = self.relu99(x2203)
        x2209 = self.pool2d28(x2204)
        x2218 = self.conv2d100(x2209)
        x2223 = self.batchnorm100(x2218)
        x2224 = self.relu100(x2223)
        x2229 = self.pool2d29(x2224)
        x2238 = self.conv2d101(x2229)
        x2243 = self.batchnorm101(x2238)
        x2244 = self.relu101(x2243)
        x2249 = self.pool2d30(x2244)
        x2258 = self.conv2d102(x2249)
        x2263 = self.batchnorm102(x2258)
        x2264 = self.relu102(x2263)
        x2269 = self.pool2d31(x2264)
        x2278 = self.conv2d103(x2269)
        x2283 = self.batchnorm103(x2278)
        x2284 = self.relu103(x2283)
        x2289 = self.pool2d32(x2284)
        x2298 = self.conv2d104(x2289)
        x2303 = self.batchnorm104(x2298)
        x2304 = self.relu104(x2303)
        x2313 = self.conv2d105(x2304)
        x2318 = self.batchnorm105(x2313)
        x2319 = self.relu105(x2318)
        x2320 = [x2319, x2304]
        print(x2319.type, x2304.type)
        x2321 = paddle.concat(x=x2320, axis=1)
        x2330 = self.conv2d106(x2321)
        x2335 = self.batchnorm106(x2330)
        x2336 = self.relu106(x2335)
        x2338 = paddle.nn.functional.interpolate(x=x2336, size=[18, 18], mode='bilinear')
        x2339 = [x2338, x2284]
        print(x2338.type, x2284.type)
        x2340 = paddle.concat(x=x2339, axis=1)
        x2349 = self.conv2d107(x2340)
        x2354 = self.batchnorm107(x2349)
        x2355 = self.relu107(x2354)
        x2357 = paddle.nn.functional.interpolate(x=x2355, size=[36, 36], mode='bilinear')
        x2358 = [x2357, x2264]
        print(x2357.type, x2264.type)
        x2359 = paddle.concat(x=x2358, axis=1)
        x2368 = self.conv2d108(x2359)
        x2373 = self.batchnorm108(x2368)
        x2374 = self.relu108(x2373)
        x2376 = paddle.nn.functional.interpolate(x=x2374, size=[72, 72], mode='bilinear')
        x2377 = [x2376, x2244]
        print(x2376.type, x2244.type)
        x2378 = paddle.concat(x=x2377, axis=1)
        x2387 = self.conv2d109(x2378)
        x2392 = self.batchnorm109(x2387)
        x2393 = self.relu109(x2392)
        x2395 = paddle.nn.functional.interpolate(x=x2393, size=[144, 144], mode='bilinear')
        x2396 = [x2395, x2224]
        print(x2395.type, x2224.type)
        x2397 = paddle.concat(x=x2396, axis=1)
        x2406 = self.conv2d110(x2397)
        x2411 = self.batchnorm110(x2406)
        x2412 = self.relu110(x2411)
        x2414 = paddle.nn.functional.interpolate(x=x2412, size=[288, 288], mode='bilinear')
        x2415 = [x2414, x2204]
        print(x2414.type, x2204.type)
        x2416 = paddle.concat(x=x2415, axis=1)
        x2425 = self.conv2d111(x2416)
        x2430 = self.batchnorm111(x2425)
        x2431 = self.relu111(x2430)
        x2432 = x2431 + x2189
        x2442 = self.conv2d112(x2432)
        x2452 = self.conv2d113(x2147)
        x2456 = paddle.nn.functional.interpolate(x=x2452, size=[288, 288], mode='bilinear')
        x2466 = self.conv2d114(x1903)
        x2470 = paddle.nn.functional.interpolate(x=x2466, size=[288, 288], mode='bilinear')
        x2480 = self.conv2d115(x1700)
        x2484 = paddle.nn.functional.interpolate(x=x2480, size=[288, 288], mode='bilinear')
        x2494 = self.conv2d116(x1538)
        x2498 = paddle.nn.functional.interpolate(x=x2494, size=[288, 288], mode='bilinear')
        x2508 = self.conv2d117(x1391)
        x2512 = paddle.nn.functional.interpolate(x=x2508, size=[288, 288], mode='bilinear')
        x2513 = [x2442, x2456, x2470, x2484, x2498, x2512]
        print(x2442.type, x2456.type, x2470.type, x2484.type, x2498.type, x2512.type)
        x2515 = paddle.concat(x=x2513, axis=1)
        x2525 = self.conv2d118(x2515)
        x2526 = (x2525, x2442, x2456, x2470, x2484, x2498, x2512)
        return x2526

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 288, 288], type-float32.
    paddle.disable_static()
    params = paddle.load(r'/home/hyj/code/u2net_nni/saves/u2net_export/u2netlbpl/u2netlbpl_web_ori_ori2_left_ori3_add1_add2_aug3_new_dreame/paddle/model.pdparams')
    model = U2NETF_LBP()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    out = model(x0)
    return out
