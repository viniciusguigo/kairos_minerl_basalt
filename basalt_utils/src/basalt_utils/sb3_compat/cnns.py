import torch
from torch import nn
from torchvision.models.resnet import BasicBlock as BasicResidualBlock
from stable_baselines3.common.preprocessing import preprocess_obs


NETWORK_ARCHITECTURE_DEFINITIONS = {
    'BasicCNN': [
            {'out_dim': 32, 'kernel_size': 8, 'stride': 4},
            {'out_dim': 64, 'kernel_size': 4, 'stride': 2},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 1},
        ],
    'MAGICALCNN': [
            {'out_dim': 32, 'kernel_size': 5, 'stride': 1, 'padding': 2},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        ],
    'MAGICALCNN-resnet': [
            {'out_dim': 64, 'stride': 4, 'residual': True},
            {'out_dim': 128, 'stride': 2, 'residual': True},
        ],
    'MAGICALCNN-resnet-128': [
            {'out_dim': 64, 'stride': 4, 'residual': True},
            {'out_dim': 128, 'stride': 2, 'residual': True},
            {'out_dim': 128, 'stride': 2, 'residual': True},
        ],
    'MAGICALCNN-resnet-256': [
            {'out_dim': 64, 'stride': 4, 'residual': True},
            {'out_dim': 128, 'stride': 2, 'residual': True},
            {'out_dim': 256, 'stride': 2, 'residual': True},
        ],
    'MAGICALCNN-small': [
            {'out_dim': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    ]
}

def compute_output_shape(observation_space, layers):
    """Compute the size of the output after passing an observation from
    `observation_space` through the given `layers`."""
    # [None] adds a batch dimension to the random observation
    torch_obs = torch.tensor(observation_space.sample()[None])
    with torch.no_grad():
        sample = preprocess_obs(torch_obs, observation_space, normalize_images=True)
        for layer in layers:
            # forward prop to compute the right size
            sample = layer(sample)

    # make sure batch axis still matches
    assert sample.shape[0] == torch_obs.shape[0]

    # return everything else
    return sample.shape[1:]


def magical_conv_block(in_chans, out_chans, kernel_size, stride, padding, use_bn, use_sn, dropout, activation_cls):
    # We sometimes disable bias because batch norm has its own bias.
    conv_layer = nn.Conv2d(
        in_chans,
        out_chans,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=not use_bn,
        padding_mode='zeros')

    if use_sn:
        # apply spectral norm if necessary
        conv_layer = nn.utils.spectral_norm(conv_layer)

    layers = [conv_layer]

    if dropout:
        # dropout after conv, but before activation
        # (doesn't matter for ReLU)
        layers.append(nn.Dropout2d(dropout))

    layers.append(activation_cls())

    if use_bn:
        # Insert BN layer after convolution (and optionally after
        # dropout). I doubt order matters much, but see here for
        # CONTROVERSY:
        # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        layers.append(nn.BatchNorm2d(out_chans))

    return layers


class MAGICALCNN(nn.Module):
    """The CNN from the MAGICAL paper."""
    def __init__(self,
                 observation_space,
                 representation_dim=128,
                 use_bn=True,
                 use_ln=False,
                 dropout=None,
                 use_sn=False,
                 arch_str='MAGICALCNN-resnet-128',
                 ActivationCls=torch.nn.ReLU):
        super().__init__()

        # If block_type == resnet, use ResNet's basic block.
        # If block_type == magical, use MAGICAL block from its paper.
        assert arch_str in NETWORK_ARCHITECTURE_DEFINITIONS.keys()
        width = 1 if 'resnet' in arch_str else 2
        self.features_dim = representation_dim
        w = width
        self.architecture_definition = NETWORK_ARCHITECTURE_DEFINITIONS[arch_str]
        conv_layers = []
        in_dim = observation_space.shape[0]

        block = magical_conv_block
        if 'resnet' in arch_str:
            block = BasicResidualBlock
        for layer_definition in self.architecture_definition:
            if layer_definition.get('residual', False):
                block_kwargs = {
                    'stride': layer_definition['stride'],
                    'downsample': nn.Sequential(nn.Conv2d(in_dim,
                                                          layer_definition['out_dim'],
                                                          kernel_size=1,
                                                          stride=layer_definition['stride']),
                                                nn.BatchNorm2d(layer_definition['out_dim']))
                }
                conv_layers += [block(in_dim,
                                      layer_definition['out_dim'] * w,
                                      **block_kwargs)]
            else:
                block_kwargs = {
                    'stride': layer_definition['stride'],
                    'kernel_size': layer_definition['kernel_size'],
                    'padding': layer_definition['padding'],
                    'use_bn': use_bn,
                    'use_sn': use_sn,
                    'dropout': dropout,
                    'activation_cls': ActivationCls
                }
                conv_layers += block(in_dim,
                                     layer_definition['out_dim'] * w,
                                     **block_kwargs)

            in_dim = layer_definition['out_dim']*w
        if 'resnet' in arch_str:
            conv_layers.append(nn.Conv2d(in_dim, 32, 1))
        conv_layers.append(nn.Flatten())

        # another FC layer to make feature maps the right size
        fc_in_size, = compute_output_shape(observation_space, conv_layers)
        fc_layers = [
            nn.Linear(fc_in_size, 128 * w),
            ActivationCls(),
            nn.Linear(128 * w, representation_dim),
        ]
        if use_sn:
            # apply SN to linear layers too
            fc_layers = [
                nn.utils.spectral_norm(layer) if isinstance(layer, nn.Linear) else layer
                for layer in fc_layers
            ]

        all_layers = [*conv_layers, *fc_layers]
        self.shared_network = nn.Sequential(*all_layers)

    def forward(self, x):
        # warn_on_non_image_tensor(x)
        return self.shared_network(x)
