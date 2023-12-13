import torch
import torch.nn as nn
import sparseconvnet as scn


class FBSparseMobileneV2(nn.Module):
    def __init__(self, nr_classes, input_channels=2, width_mult=1.0, size_before_avg_pool=[5,7]):
        super(FBSparseMobileneV2, self).__init__()
        cnn_spatial_output_size = [1, 1]
        cnn_spatial_size_before_avg_pool = size_before_avg_pool
        sparse_out_channels = int(1280 * width_mult)
        self.sparseModel = SparseMobileNet(2, nInputPlanes=input_channels, layers=[
            [16, int(width_mult * 32), 1, 2],
            [1,  int(width_mult * 16), 1, 1],
            [6,  int(width_mult * 24), 2, 2],
            [6,  int(width_mult * 32), 3, 2],
            [6,  int(width_mult * 64), 4, 2],
            [6,  int(width_mult * 96), 3, 1],
            [6,  int(width_mult * 160), 3, 2],
            [6,  int(width_mult * 320), 1, 1],
            # [6, sparse_out_channels, 1, 2]
            ]
        ).add(scn.SubmanifoldConvolution(2, int(width_mult * 320), sparse_out_channels, 1,False)
        ).add(scn.BatchNormReLU(sparse_out_channels)
        ).add(scn.AveragePooling(dimension=2, pool_size=cnn_spatial_size_before_avg_pool, pool_stride=1)
        ).add(scn.SparseToDense(2, sparse_out_channels))
        
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = sparse_out_channels #cnn_spatial_output_size[0] * cnn_spatial_output_size[1] * sparse_out_channels
        self.linear = nn.Linear(self.linear_input_features, nr_classes)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear(x)

        return x


def SparseMobileNet(dimension, nInputPlanes, layers):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    ic = nInputPlanes
    m = scn.Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()

    def InvertedResidual(nIn, nOut, t, stride):
        hidden_dim = int(round(nIn * t))
        use_res = stride == 1 and nIn == nOut
        if t == 1:
            return scn.Sequential().add(
                        scn.SubmanifoldConvolution(
                            dimension,
                            hidden_dim,
                            hidden_dim,
                            3,
                            False, 
                            groups=hidden_dim)).add(
                        scn.BatchNormReLU(hidden_dim)).add(
                        scn.SubmanifoldConvolution(
                            dimension,
                            hidden_dim,
                            nOut,
                            1,
                            False)
                        ).add(
                        scn.BatchNormalization(nOut)
                        )
        elif use_res:
            return  scn.Sequential().add(
                    scn.ConcatTable().add(
                    scn.Sequential().add(
                    scn.SubmanifoldConvolution(
                        dimension,
                        nIn,
                        hidden_dim,
                        1,
                        False)).add(
                    scn.BatchNormReLU(hidden_dim)).add(
                    scn.SubmanifoldConvolution(
                        dimension,
                        hidden_dim,
                        hidden_dim,
                        3,
                        False, 
                        groups=hidden_dim)
                    ).add(
                    scn.BatchNormReLU(hidden_dim)).add(
                    scn.SubmanifoldConvolution(
                        dimension,
                        hidden_dim,
                        nOut,
                        1,
                        False)
                    ).add(
                    scn.BatchNormalization(nOut)
                    )
                    ).add(scn.Identity())
                    ).add(scn.AddTable())

        elif stride == 2:
            return  scn.Sequential().add(
                        scn.SubmanifoldConvolution(
                            dimension,
                            nIn,
                            hidden_dim,
                            1,
                            False)).add(
                        scn.BatchNormReLU(hidden_dim)).add(
                        scn.Convolution(
                            dimension,
                            hidden_dim,
                            hidden_dim,
                            3,
                            filter_stride=2,
                            bias=False, 
                            groups=hidden_dim)
                        ).add(
                        scn.BatchNormReLU(hidden_dim)).add(
                        scn.SubmanifoldConvolution(
                            dimension,
                            hidden_dim,
                            nOut,
                            1,
                            False)
                        ).add(
                        scn.BatchNormalization(nOut)
                        )
        else:
            return  scn.Sequential().add(
            scn.SubmanifoldConvolution(
                dimension,
                nIn,
                hidden_dim,
                1,
                False)).add(
            scn.BatchNormReLU(hidden_dim)).add(
            scn.SubmanifoldConvolution(
                dimension,
                hidden_dim,
                hidden_dim,
                3,
                bias=False, 
                groups=hidden_dim)
            ).add(
            scn.BatchNormReLU(hidden_dim)).add(
            scn.SubmanifoldConvolution(
                dimension,
                hidden_dim,
                nOut,
                1,
                False)
            ).add(
            scn.BatchNormalization(nOut)
            )
                    

    for t, c, reps, s in layers:
        oc = c
        for rep in range(reps):
            stride = s if (rep == 0) else 1
            m.add(InvertedResidual(ic, oc, t, stride))
            ic = oc

    return m