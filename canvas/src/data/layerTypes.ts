import { LayerType } from '../types';

export const layerTypes: LayerType[] = [
  // ── Input/Output Nodes ──────────────────────────────────────────────────────
  {
    id: 'input',
    name: 'Input',
    displayName: 'Input',
    description: 'Model input node',
    icon: 'ArrowRight',
    category: 'io',
    parameters: [
      {
        name: 'shape',
        type: 'string',
        defaultValue: '1,3,224,224',
        required: true,
        description: 'Input tensor shape (comma-separated, e.g., 1,3,224,224)'
      },
      {
        name: 'name',
        type: 'string',
        defaultValue: 'input',
        required: false,
        description: 'Name for this input'
      }
    ]
  },

  {
    id: 'output',
    name: 'Output',
    displayName: 'Output',
    description: 'Model output node',
    icon: 'ArrowLeft',
    category: 'io',
    parameters: [
      {
        name: 'name',
        type: 'string',
        defaultValue: 'output',
        required: false,
        description: 'Name for this output'
      }
    ]
  },

  // ── Module/Group Nodes ───────────────────────────────────────────────────────
  {
    id: 'module',
    name: 'Module',
    displayName: 'Module',
    description: 'Custom module group (nn.Module)',
    icon: 'Package',
    category: 'module',
    parameters: [
      {
        name: 'name',
        type: 'string',
        defaultValue: 'CustomModule',
        required: true,
        description: 'Module class name'
      },
      {
        name: 'description',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Module description'
      }
    ]
  },

  // ── Convolutional Layers ─────────────────────────────────────────────────────
  {
    id: 'conv1d',
    name: 'Conv1d',
    displayName: 'Conv1D',
    description: '1D Convolutional Layer',
    icon: 'Minus',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 1, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to both sides', min: 0 },
      { name: 'dilation', type: 'number', defaultValue: 1, required: false, description: 'Spacing between kernel elements', min: 1 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections from input to output channels', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'conv2d',
    name: 'Conv2d',
    displayName: 'Conv2D',
    description: '2D Convolutional Layer',
    icon: 'Grid3X3',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 3, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to both sides', min: 0 },
      { name: 'dilation', type: 'number', defaultValue: 1, required: false, description: 'Spacing between kernel elements', min: 1 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections from input to output channels', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'conv3d',
    name: 'Conv3d',
    displayName: 'Conv3D',
    description: '3D Convolutional Layer',
    icon: 'Box',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 1, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to all sides', min: 0 },
      { name: 'dilation', type: 'number', defaultValue: 1, required: false, description: 'Spacing between kernel elements', min: 1 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections from input to output channels', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'convtranspose1d',
    name: 'ConvTranspose1d',
    displayName: 'ConvTranspose1D',
    description: '1D Transposed Convolutional Layer',
    icon: 'ChevronsRight',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 32, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to input', min: 0 },
      { name: 'output_padding', type: 'number', defaultValue: 0, required: false, description: 'Additional size added to one side of the output shape', min: 0 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'convtranspose2d',
    name: 'ConvTranspose2d',
    displayName: 'ConvTranspose2D',
    description: '2D Transposed Convolutional Layer',
    icon: 'ChevronsRight',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 32, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to input', min: 0 },
      { name: 'output_padding', type: 'number', defaultValue: 0, required: false, description: 'Additional size added to one side of the output shape', min: 0 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'convtranspose3d',
    name: 'ConvTranspose3d',
    displayName: 'ConvTranspose3D',
    description: '3D Transposed Convolutional Layer',
    icon: 'ChevronsRight',
    category: 'conv',
    parameters: [
      { name: 'in_channels', type: 'number', defaultValue: 64, required: true, description: 'Number of input channels', min: 1 },
      { name: 'out_channels', type: 'number', defaultValue: 32, required: true, description: 'Number of output channels', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of the convolving kernel', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride of the convolution', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to input', min: 0 },
      { name: 'output_padding', type: 'number', defaultValue: 0, required: false, description: 'Additional size added to one side of the output shape', min: 0 },
      { name: 'groups', type: 'number', defaultValue: 1, required: false, description: 'Number of blocked connections', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  // ── Linear / Projection Layers ────────────────────────────────────────────────
  {
    id: 'linear',
    name: 'Linear',
    displayName: 'Linear',
    description: 'Fully Connected Layer',
    icon: 'Minus',
    category: 'linear',
    parameters: [
      { name: 'in_features', type: 'number', defaultValue: 512, required: true, description: 'Size of input features', min: 1 },
      { name: 'out_features', type: 'number', defaultValue: 256, required: true, description: 'Size of output features', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If set to False, the layer will not learn an additive bias' }
    ]
  },

  {
    id: 'bilinear',
    name: 'Bilinear',
    displayName: 'Bilinear',
    description: 'Bilinear transformation of two inputs: y = x1^T A x2 + b',
    icon: 'GitMerge',
    category: 'linear',
    parameters: [
      { name: 'in1_features', type: 'number', defaultValue: 128, required: true, description: 'Size of first input', min: 1 },
      { name: 'in2_features', type: 'number', defaultValue: 128, required: true, description: 'Size of second input', min: 1 },
      { name: 'out_features', type: 'number', defaultValue: 64, required: true, description: 'Size of output', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  {
    id: 'identity',
    name: 'Identity',
    displayName: 'Identity',
    description: 'Placeholder identity operator that is argument-insensitive',
    icon: 'ArrowRight',
    category: 'linear',
    parameters: []
  },

  {
    id: 'lazylinear',
    name: 'LazyLinear',
    displayName: 'LazyLinear',
    description: 'Linear layer with lazy initialization of in_features',
    icon: 'Zap',
    category: 'linear',
    parameters: [
      { name: 'out_features', type: 'number', defaultValue: 256, required: true, description: 'Size of output features', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds a learnable bias' }
    ]
  },

  // ── Activation Functions ──────────────────────────────────────────────────────
  {
    id: 'relu',
    name: 'ReLU',
    displayName: 'ReLU',
    description: 'Rectified Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'relu6',
    name: 'ReLU6',
    displayName: 'ReLU6',
    description: 'ReLU capped at 6: min(max(0, x), 6)',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'leakyrelu',
    name: 'LeakyReLU',
    displayName: 'LeakyReLU',
    description: 'Leaky Rectified Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'negative_slope', type: 'number', defaultValue: 0.01, required: false, description: 'Controls the angle of the negative slope', min: 0 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'prelu',
    name: 'PReLU',
    displayName: 'PReLU',
    description: 'Parametric Rectified Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'num_parameters', type: 'number', defaultValue: 1, required: false, description: 'Number of learnable parameters', min: 1 },
      { name: 'init', type: 'number', defaultValue: 0.25, required: false, description: 'Initial value of the learnable weight', min: 0 }
    ]
  },

  {
    id: 'elu',
    name: 'ELU',
    displayName: 'ELU',
    description: 'Exponential Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'alpha', type: 'number', defaultValue: 1.0, required: false, description: 'The alpha value for the ELU formulation', min: 0 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'celu',
    name: 'CELU',
    displayName: 'CELU',
    description: 'Continuously Differentiable Exponential Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'alpha', type: 'number', defaultValue: 1.0, required: false, description: 'The alpha value for the CELU formulation', min: 0 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'selu',
    name: 'SELU',
    displayName: 'SELU',
    description: 'Scaled Exponential Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'gelu',
    name: 'GELU',
    displayName: 'GELU',
    description: 'Gaussian Error Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      {
        name: 'approximate',
        type: 'select',
        defaultValue: 'none',
        required: false,
        description: "The gelu approximation algorithm to use: 'none' or 'tanh'",
        options: ['none', 'tanh']
      }
    ]
  },

  {
    id: 'silu',
    name: 'SiLU',
    displayName: 'SiLU',
    description: 'Sigmoid Linear Unit (Swish)',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'mish',
    name: 'Mish',
    displayName: 'Mish',
    description: 'Mish activation: x * tanh(softplus(x))',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'sigmoid',
    name: 'Sigmoid',
    displayName: 'Sigmoid',
    description: 'Sigmoid activation function',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: []
  },

  {
    id: 'hardsigmoid',
    name: 'Hardsigmoid',
    displayName: 'Hardsigmoid',
    description: 'Hard sigmoid activation',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'tanh',
    name: 'Tanh',
    displayName: 'Tanh',
    description: 'Hyperbolic tangent activation',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: []
  },

  {
    id: 'hardtanh',
    name: 'Hardtanh',
    displayName: 'Hardtanh',
    description: 'Hard tanh activation, clips input to [min_val, max_val]',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'min_val', type: 'number', defaultValue: -1.0, required: false, description: 'Minimum value of the linear region' },
      { name: 'max_val', type: 'number', defaultValue: 1.0, required: false, description: 'Maximum value of the linear region' },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'hardswish',
    name: 'Hardswish',
    displayName: 'Hardswish',
    description: 'Hard Swish activation',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'softmax',
    name: 'Softmax',
    displayName: 'Softmax',
    description: 'Softmax over a dimension',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'dim', type: 'number', defaultValue: 1, required: true, description: 'Dimension along which Softmax is computed' }
    ]
  },

  {
    id: 'logsoftmax',
    name: 'LogSoftmax',
    displayName: 'LogSoftmax',
    description: 'Log-Softmax over a dimension',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'dim', type: 'number', defaultValue: 1, required: true, description: 'Dimension along which LogSoftmax is computed' }
    ]
  },

  {
    id: 'softplus',
    name: 'Softplus',
    displayName: 'Softplus',
    description: 'Softplus activation: (1/beta) * log(1 + exp(beta * x))',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'beta', type: 'number', defaultValue: 1, required: false, description: 'The beta value for the Softplus formulation', min: 0 },
      { name: 'threshold', type: 'number', defaultValue: 20, required: false, description: 'Values above this revert to a linear function', min: 0 }
    ]
  },

  {
    id: 'softsign',
    name: 'Softsign',
    displayName: 'Softsign',
    description: 'Softsign activation: x / (1 + |x|)',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: []
  },

  {
    id: 'tanhshrink',
    name: 'Tanhshrink',
    displayName: 'Tanhshrink',
    description: 'Tanhshrink: x - tanh(x)',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: []
  },

  {
    id: 'softshrink',
    name: 'Softshrink',
    displayName: 'Softshrink',
    description: 'Soft shrinkage function',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'lambd', type: 'number', defaultValue: 0.5, required: false, description: 'The lambda value for the Softshrink formulation', min: 0 }
    ]
  },

  {
    id: 'hardshrink',
    name: 'Hardshrink',
    displayName: 'Hardshrink',
    description: 'Hard shrinkage function',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'lambd', type: 'number', defaultValue: 0.5, required: false, description: 'The lambda value for the Hardshrink formulation', min: 0 }
    ]
  },

  {
    id: 'rrelu',
    name: 'RReLU',
    displayName: 'RReLU',
    description: 'Randomized Leaky ReLU',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'lower', type: 'number', defaultValue: 0.125, required: false, description: 'Lower bound of the uniform distribution', min: 0, max: 1 },
      { name: 'upper', type: 'number', defaultValue: 0.3333, required: false, description: 'Upper bound of the uniform distribution', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'threshold',
    name: 'Threshold',
    displayName: 'Threshold',
    description: 'Thresholds each element of the input tensor',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'threshold', type: 'number', defaultValue: 0.0, required: true, description: 'The value to threshold at' },
      { name: 'value', type: 'number', defaultValue: 0.0, required: true, description: 'The value to replace with' },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'Can optionally do the operation in-place' }
    ]
  },

  {
    id: 'glu',
    name: 'GLU',
    displayName: 'GLU',
    description: 'Gated Linear Unit',
    icon: 'TrendingUp',
    category: 'activation',
    parameters: [
      { name: 'dim', type: 'number', defaultValue: -1, required: false, description: 'Dimension on which to split the input' }
    ]
  },

  // ── Pooling Layers ───────────────────────────────────────────────────────────
  {
    id: 'maxpool1d',
    name: 'MaxPool1d',
    displayName: 'MaxPool1D',
    description: '1D Max Pooling',
    icon: 'ChevronUp',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding', min: 0 }
    ]
  },

  {
    id: 'maxpool2d',
    name: 'MaxPool2d',
    displayName: 'MaxPool2D',
    description: '2D Max Pooling',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window to take a max over', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding to be added', min: 0 }
    ]
  },

  {
    id: 'maxpool3d',
    name: 'MaxPool3d',
    displayName: 'MaxPool3D',
    description: '3D Max Pooling',
    icon: 'Box',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding', min: 0 }
    ]
  },

  {
    id: 'avgpool1d',
    name: 'AvgPool1d',
    displayName: 'AvgPool1D',
    description: '1D Average Pooling',
    icon: 'ChevronUp',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding', min: 0 }
    ]
  },

  {
    id: 'avgpool2d',
    name: 'AvgPool2d',
    displayName: 'AvgPool2D',
    description: '2D Average Pooling',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding to be added', min: 0 }
    ]
  },

  {
    id: 'avgpool3d',
    name: 'AvgPool3d',
    displayName: 'AvgPool3D',
    description: '3D Average Pooling',
    icon: 'Box',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Implicit zero padding', min: 0 }
    ]
  },

  {
    id: 'adaptivemaxpool1d',
    name: 'AdaptiveMaxPool1d',
    displayName: 'AdaptiveMaxPool1D',
    description: '1D Adaptive Max Pooling to a target output size',
    icon: 'ChevronUp',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size', min: 1 }
    ]
  },

  {
    id: 'adaptivemaxpool2d',
    name: 'AdaptiveMaxPool2d',
    displayName: 'AdaptiveMaxPool2D',
    description: '2D Adaptive Max Pooling to a target output size',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size (H × W)', min: 1 }
    ]
  },

  {
    id: 'adaptivemaxpool3d',
    name: 'AdaptiveMaxPool3d',
    displayName: 'AdaptiveMaxPool3D',
    description: '3D Adaptive Max Pooling to a target output size',
    icon: 'Box',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size (D × H × W)', min: 1 }
    ]
  },

  {
    id: 'adaptiveavgpool1d',
    name: 'AdaptiveAvgPool1d',
    displayName: 'AdaptiveAvgPool1D',
    description: '1D Adaptive Average Pooling to a target output size',
    icon: 'ChevronUp',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size', min: 1 }
    ]
  },

  {
    id: 'adaptiveavgpool2d',
    name: 'AdaptiveAvgPool2d',
    displayName: 'AdaptiveAvgPool2D',
    description: '2D Adaptive Average Pooling to a target output size',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size (H × W)', min: 1 }
    ]
  },

  {
    id: 'adaptiveavgpool3d',
    name: 'AdaptiveAvgPool3d',
    displayName: 'AdaptiveAvgPool3D',
    description: '3D Adaptive Average Pooling to a target output size',
    icon: 'Box',
    category: 'pool',
    parameters: [
      { name: 'output_size', type: 'number', defaultValue: 1, required: true, description: 'Target output size (D × H × W)', min: 1 }
    ]
  },

  {
    id: 'fractionalmaxpool2d',
    name: 'FractionalMaxPool2d',
    displayName: 'FractionalMaxPool2D',
    description: 'Fractional Max Pooling over a 2D input',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the kernel', min: 1 },
      { name: 'output_size', type: 'number', defaultValue: 1, required: false, description: 'Target output size', min: 1 }
    ]
  },

  {
    id: 'lppool1d',
    name: 'LPPool1d',
    displayName: 'LPPool1D',
    description: '1D Power-Average Pooling',
    icon: 'ChevronUp',
    category: 'pool',
    parameters: [
      { name: 'norm_type', type: 'number', defaultValue: 2, required: true, description: 'Exponent in the norm', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 }
    ]
  },

  {
    id: 'lppool2d',
    name: 'LPPool2d',
    displayName: 'LPPool2D',
    description: '2D Power-Average Pooling',
    icon: 'Square',
    category: 'pool',
    parameters: [
      { name: 'norm_type', type: 'number', defaultValue: 2, required: true, description: 'Exponent in the norm', min: 1 },
      { name: 'kernel_size', type: 'number', defaultValue: 2, required: true, description: 'Size of the window', min: 1 },
      { name: 'stride', type: 'number', defaultValue: 2, required: false, description: 'Stride of the window', min: 1 }
    ]
  },

  // ── Normalization Layers ─────────────────────────────────────────────────────
  {
    id: 'batchnorm1d',
    name: 'BatchNorm1d',
    displayName: 'BatchNorm1D',
    description: '1D Batch Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'momentum', type: 'number', defaultValue: 0.1, required: false, description: 'Momentum for running stats', min: 0, max: 1 },
      { name: 'affine', type: 'boolean', defaultValue: true, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'batchnorm2d',
    name: 'BatchNorm2d',
    displayName: 'BatchNorm2D',
    description: '2D Batch Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features from an expected input', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'A value added to the denominator for numerical stability', min: 0 },
      { name: 'momentum', type: 'number', defaultValue: 0.1, required: false, description: 'The value used for the running_mean and running_var computation', min: 0, max: 1 },
      { name: 'affine', type: 'boolean', defaultValue: true, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'batchnorm3d',
    name: 'BatchNorm3d',
    displayName: 'BatchNorm3D',
    description: '3D Batch Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'momentum', type: 'number', defaultValue: 0.1, required: false, description: 'Momentum for running stats', min: 0, max: 1 },
      { name: 'affine', type: 'boolean', defaultValue: true, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'layernorm',
    name: 'LayerNorm',
    displayName: 'LayerNorm',
    description: 'Layer Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'normalized_shape', type: 'string', defaultValue: '512', required: true, description: 'Input shape for normalization (comma-separated for multi-dim, e.g., 512)' },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'elementwise_affine', type: 'boolean', defaultValue: true, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'groupnorm',
    name: 'GroupNorm',
    displayName: 'GroupNorm',
    description: 'Group Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_groups', type: 'number', defaultValue: 32, required: true, description: 'Number of groups for normalization', min: 1 },
      { name: 'num_channels', type: 'number', defaultValue: 256, required: true, description: 'Number of channels in input', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'affine', type: 'boolean', defaultValue: true, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'instancenorm1d',
    name: 'InstanceNorm1d',
    displayName: 'InstanceNorm1D',
    description: '1D Instance Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'affine', type: 'boolean', defaultValue: false, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'instancenorm2d',
    name: 'InstanceNorm2d',
    displayName: 'InstanceNorm2D',
    description: '2D Instance Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'affine', type: 'boolean', defaultValue: false, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'instancenorm3d',
    name: 'InstanceNorm3d',
    displayName: 'InstanceNorm3D',
    description: '3D Instance Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'num_features', type: 'number', defaultValue: 64, required: true, description: 'Number of features', min: 1 },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 },
      { name: 'affine', type: 'boolean', defaultValue: false, required: false, description: 'If True, learnable affine parameters' }
    ]
  },

  {
    id: 'rmsnorm',
    name: 'RMSNorm',
    displayName: 'RMSNorm',
    description: 'Root Mean Square Layer Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'normalized_shape', type: 'string', defaultValue: '512', required: true, description: 'Input shape for normalization (comma-separated)' },
      { name: 'eps', type: 'number', defaultValue: 1e-5, required: false, description: 'Value for numerical stability', min: 0 }
    ]
  },

  {
    id: 'localresponsenorm',
    name: 'LocalResponseNorm',
    displayName: 'LocalResponseNorm',
    description: 'Local Response Normalization',
    icon: 'Layers',
    category: 'norm',
    parameters: [
      { name: 'size', type: 'number', defaultValue: 5, required: true, description: 'Amount of neighboring channels to normalize over', min: 1 },
      { name: 'alpha', type: 'number', defaultValue: 0.0001, required: false, description: 'Multiplicative factor', min: 0 },
      { name: 'beta', type: 'number', defaultValue: 0.75, required: false, description: 'Exponent', min: 0 },
      { name: 'k', type: 'number', defaultValue: 1.0, required: false, description: 'Additive factor', min: 0 }
    ]
  },

  // ── Dropout / Regularization ─────────────────────────────────────────────────
  {
    id: 'dropout',
    name: 'Dropout',
    displayName: 'Dropout',
    description: 'Randomly zeroes some elements of the input tensor during training',
    icon: 'X',
    category: 'dropout',
    parameters: [
      { name: 'p', type: 'number', defaultValue: 0.5, required: false, description: 'Probability of an element to be zeroed', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'If set to True, will do this operation in-place' }
    ]
  },

  {
    id: 'dropout1d',
    name: 'Dropout1d',
    displayName: 'Dropout1D',
    description: 'Randomly zeroes entire channels of a 2D or 3D input',
    icon: 'X',
    category: 'dropout',
    parameters: [
      { name: 'p', type: 'number', defaultValue: 0.5, required: false, description: 'Probability of a channel to be zeroed', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'If set to True, will do this operation in-place' }
    ]
  },

  {
    id: 'dropout2d',
    name: 'Dropout2d',
    displayName: 'Dropout2D',
    description: 'Randomly zeroes entire channels of a 4D input',
    icon: 'X',
    category: 'dropout',
    parameters: [
      { name: 'p', type: 'number', defaultValue: 0.5, required: false, description: 'Probability of a channel to be zeroed', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'If set to True, will do this operation in-place' }
    ]
  },

  {
    id: 'dropout3d',
    name: 'Dropout3d',
    displayName: 'Dropout3D',
    description: 'Randomly zeroes entire channels of a 5D input',
    icon: 'X',
    category: 'dropout',
    parameters: [
      { name: 'p', type: 'number', defaultValue: 0.5, required: false, description: 'Probability of a channel to be zeroed', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'If set to True, will do this operation in-place' }
    ]
  },

  {
    id: 'alphadropout',
    name: 'AlphaDropout',
    displayName: 'AlphaDropout',
    description: 'Dropout that maintains mean and variance for SELU networks',
    icon: 'X',
    category: 'dropout',
    parameters: [
      { name: 'p', type: 'number', defaultValue: 0.5, required: false, description: 'Probability of an element to be zeroed', min: 0, max: 1 },
      { name: 'inplace', type: 'boolean', defaultValue: false, required: false, description: 'If set to True, will do this operation in-place' }
    ]
  },

  // ── Recurrent Layers ─────────────────────────────────────────────────────────
  {
    id: 'rnn',
    name: 'RNN',
    displayName: 'RNN',
    description: 'Elman Recurrent Neural Network',
    icon: 'RefreshCw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'num_layers', type: 'number', defaultValue: 1, required: false, description: 'Number of recurrent layers', min: 1 },
      { name: 'nonlinearity', type: 'select', defaultValue: 'tanh', required: false, description: 'Nonlinearity to use', options: ['tanh', 'relu'] },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' },
      { name: 'dropout', type: 'number', defaultValue: 0.0, required: false, description: 'Dropout probability between layers', min: 0, max: 1 },
      { name: 'bidirectional', type: 'boolean', defaultValue: false, required: false, description: 'If True, bidirectional RNN' }
    ]
  },

  {
    id: 'lstm',
    name: 'LSTM',
    displayName: 'LSTM',
    description: 'Long Short-Term Memory network',
    icon: 'RefreshCw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'num_layers', type: 'number', defaultValue: 1, required: false, description: 'Number of recurrent layers', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' },
      { name: 'dropout', type: 'number', defaultValue: 0.0, required: false, description: 'Dropout probability between layers', min: 0, max: 1 },
      { name: 'bidirectional', type: 'boolean', defaultValue: false, required: false, description: 'If True, bidirectional LSTM' },
      { name: 'proj_size', type: 'number', defaultValue: 0, required: false, description: 'If > 0, will use LSTM with projections', min: 0 }
    ]
  },

  {
    id: 'gru',
    name: 'GRU',
    displayName: 'GRU',
    description: 'Gated Recurrent Unit network',
    icon: 'RefreshCw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'num_layers', type: 'number', defaultValue: 1, required: false, description: 'Number of recurrent layers', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' },
      { name: 'dropout', type: 'number', defaultValue: 0.0, required: false, description: 'Dropout probability between layers', min: 0, max: 1 },
      { name: 'bidirectional', type: 'boolean', defaultValue: false, required: false, description: 'If True, bidirectional GRU' }
    ]
  },

  {
    id: 'rnncell',
    name: 'RNNCell',
    displayName: 'RNNCell',
    description: 'Single Elman RNN cell',
    icon: 'RefreshCcw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' },
      { name: 'nonlinearity', type: 'select', defaultValue: 'tanh', required: false, description: 'Nonlinearity to use', options: ['tanh', 'relu'] }
    ]
  },

  {
    id: 'lstmcell',
    name: 'LSTMCell',
    displayName: 'LSTMCell',
    description: 'Single LSTM cell',
    icon: 'RefreshCcw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' }
    ]
  },

  {
    id: 'grucell',
    name: 'GRUCell',
    displayName: 'GRUCell',
    description: 'Single GRU cell',
    icon: 'RefreshCcw',
    category: 'recurrent',
    parameters: [
      { name: 'input_size', type: 'number', defaultValue: 128, required: true, description: 'Number of input features', min: 1 },
      { name: 'hidden_size', type: 'number', defaultValue: 256, required: true, description: 'Number of features in the hidden state', min: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If False, no bias weights' }
    ]
  },

  // ── Embedding Layers ─────────────────────────────────────────────────────────
  {
    id: 'embedding',
    name: 'Embedding',
    displayName: 'Embedding',
    description: 'Lookup table for dense vector embeddings',
    icon: 'Hash',
    category: 'embedding',
    parameters: [
      { name: 'num_embeddings', type: 'number', defaultValue: 10000, required: true, description: 'Size of the dictionary of embeddings', min: 1 },
      { name: 'embedding_dim', type: 'number', defaultValue: 128, required: true, description: 'Size of each embedding vector', min: 1 },
      { name: 'padding_idx', type: 'number', defaultValue: 0, required: false, description: 'Pad output with zeros at this index', min: 0 },
      { name: 'max_norm', type: 'number', defaultValue: 0, required: false, description: 'If > 0, re-normalizes embeddings with norm > max_norm', min: 0 },
      { name: 'sparse', type: 'boolean', defaultValue: false, required: false, description: 'If True, gradient w.r.t. weight is sparse' }
    ]
  },

  {
    id: 'embeddingbag',
    name: 'EmbeddingBag',
    displayName: 'EmbeddingBag',
    description: 'Computes sum or mean of embeddings without instantiating intermediate tensors',
    icon: 'Hash',
    category: 'embedding',
    parameters: [
      { name: 'num_embeddings', type: 'number', defaultValue: 10000, required: true, description: 'Size of the dictionary of embeddings', min: 1 },
      { name: 'embedding_dim', type: 'number', defaultValue: 128, required: true, description: 'Size of each embedding vector', min: 1 },
      { name: 'mode', type: 'select', defaultValue: 'mean', required: false, description: 'Aggregation mode', options: ['sum', 'mean', 'max'] },
      { name: 'sparse', type: 'boolean', defaultValue: false, required: false, description: 'If True, gradient w.r.t. weight is sparse' }
    ]
  },

  // ── Tensor Manipulation ──────────────────────────────────────────────────────
  {
    id: 'flatten',
    name: 'Flatten',
    displayName: 'Flatten',
    description: 'Flattens input tensor contiguously',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'start_dim', type: 'number', defaultValue: 1, required: false, description: 'First dim to flatten', min: 0 },
      { name: 'end_dim', type: 'number', defaultValue: -1, required: false, description: 'Last dim to flatten' }
    ]
  },

  {
    id: 'unflatten',
    name: 'Unflatten',
    displayName: 'Unflatten',
    description: 'Expands a tensor dimension into a new shape',
    icon: 'Minimize2',
    category: 'tensor',
    parameters: [
      { name: 'dim', type: 'number', defaultValue: 1, required: true, description: 'Dimension to unflatten' },
      { name: 'unflattened_size', type: 'string', defaultValue: '4,4', required: true, description: 'New shape for the dimension (comma-separated)' }
    ]
  },

  {
    id: 'upsample',
    name: 'Upsample',
    displayName: 'Upsample',
    description: 'Upsamples input to the given size or scale factor',
    icon: 'ZoomIn',
    category: 'tensor',
    parameters: [
      { name: 'scale_factor', type: 'number', defaultValue: 2, required: false, description: 'Multiplier for spatial size', min: 1 },
      {
        name: 'mode',
        type: 'select',
        defaultValue: 'nearest',
        required: false,
        description: 'Upsampling algorithm',
        options: ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
      },
      // Same conditional as interpolate: torch raises for align_corners with
      // mode='nearest', so the field is hidden unless the mode can use it.
      { name: 'align_corners', type: 'boolean', defaultValue: false, required: false, description: 'If True, aligns corners of input and output tensors', showIf: { param: 'mode', oneOf: ['linear', 'bilinear', 'bicubic', 'trilinear'] } }
    ]
  },

  {
    id: 'pixelshuffle',
    name: 'PixelShuffle',
    displayName: 'PixelShuffle',
    description: 'Rearranges elements in a tensor (sub-pixel convolution)',
    icon: 'Grid',
    category: 'tensor',
    parameters: [
      { name: 'upscale_factor', type: 'number', defaultValue: 2, required: true, description: 'Factor to increase spatial resolution by', min: 1 }
    ]
  },

  {
    id: 'pixelunshuffle',
    name: 'PixelUnshuffle',
    displayName: 'PixelUnshuffle',
    description: 'Reverses PixelShuffle by rearranging spatial elements into depth',
    icon: 'Grid',
    category: 'tensor',
    parameters: [
      { name: 'downscale_factor', type: 'number', defaultValue: 2, required: true, description: 'Factor to reduce spatial resolution by', min: 1 }
    ]
  },

  {
    id: 'channelshuffle',
    name: 'ChannelShuffle',
    displayName: 'ChannelShuffle',
    description: 'Shuffle channels in a tensor (ShuffleNet)',
    icon: 'Shuffle',
    category: 'tensor',
    parameters: [
      { name: 'groups', type: 'number', defaultValue: 3, required: true, description: 'Number of groups to divide channels in', min: 1 }
    ]
  },

  {
    id: 'zeropad2d',
    name: 'ZeroPad2d',
    displayName: 'ZeroPad2D',
    description: 'Pads a 4D tensor with zeros',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding on all sides', min: 0 }
    ]
  },

  {
    id: 'constantpad1d',
    name: 'ConstantPad1d',
    displayName: 'ConstantPad1D',
    description: 'Pads a 3D tensor with a constant value',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 },
      { name: 'value', type: 'number', defaultValue: 0, required: true, description: 'Fill value' }
    ]
  },

  {
    id: 'constantpad2d',
    name: 'ConstantPad2d',
    displayName: 'ConstantPad2D',
    description: 'Pads a 4D tensor with a constant value',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 },
      { name: 'value', type: 'number', defaultValue: 0, required: true, description: 'Fill value' }
    ]
  },

  {
    id: 'constantpad3d',
    name: 'ConstantPad3d',
    displayName: 'ConstantPad3D',
    description: 'Pads a 5D tensor with a constant value',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 },
      { name: 'value', type: 'number', defaultValue: 0, required: true, description: 'Fill value' }
    ]
  },

  {
    id: 'reflectionpad1d',
    name: 'ReflectionPad1d',
    displayName: 'ReflectionPad1D',
    description: 'Pads a 3D tensor using reflection of input boundary',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'reflectionpad2d',
    name: 'ReflectionPad2d',
    displayName: 'ReflectionPad2D',
    description: 'Pads a 4D tensor using reflection of input boundary',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'reflectionpad3d',
    name: 'ReflectionPad3d',
    displayName: 'ReflectionPad3D',
    description: 'Pads a 5D tensor using reflection of input boundary',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'replicationpad1d',
    name: 'ReplicationPad1d',
    displayName: 'ReplicationPad1D',
    description: 'Pads a 3D tensor by replicating boundary values',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'replicationpad2d',
    name: 'ReplicationPad2d',
    displayName: 'ReplicationPad2D',
    description: 'Pads a 4D tensor by replicating boundary values',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'replicationpad3d',
    name: 'ReplicationPad3d',
    displayName: 'ReplicationPad3D',
    description: 'Pads a 5D tensor by replicating boundary values',
    icon: 'Maximize2',
    category: 'tensor',
    parameters: [
      { name: 'padding', type: 'number', defaultValue: 1, required: true, description: 'Size of padding', min: 0 }
    ]
  },

  {
    id: 'fold',
    name: 'Fold',
    displayName: 'Fold',
    description: 'Combines an array of sliding blocks into a large containing tensor',
    icon: 'Layers',
    category: 'tensor',
    parameters: [
      { name: 'output_size', type: 'string', defaultValue: '8,8', required: true, description: 'Shape of the spatial dimensions of the output (H,W)' },
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of each block', min: 1 },
      { name: 'dilation', type: 'number', defaultValue: 1, required: false, description: 'Spacing between kernel elements', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to both sides', min: 0 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride between blocks', min: 1 }
    ]
  },

  {
    id: 'unfold',
    name: 'Unfold',
    displayName: 'Unfold',
    description: 'Extracts sliding local blocks from a batched input tensor',
    icon: 'Layers',
    category: 'tensor',
    parameters: [
      { name: 'kernel_size', type: 'number', defaultValue: 3, required: true, description: 'Size of each block', min: 1 },
      { name: 'dilation', type: 'number', defaultValue: 1, required: false, description: 'Spacing between kernel elements', min: 1 },
      { name: 'padding', type: 'number', defaultValue: 0, required: false, description: 'Zero-padding added to both sides', min: 0 },
      { name: 'stride', type: 'number', defaultValue: 1, required: false, description: 'Stride between blocks', min: 1 }
    ]
  },

  // ── Loss Functions ───────────────────────────────────────────────────────────
  {
    id: 'crossentropyloss',
    name: 'CrossEntropyLoss',
    displayName: 'CrossEntropyLoss',
    description: 'Cross entropy loss combining LogSoftmax and NLLLoss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      },
      { name: 'label_smoothing', type: 'number', defaultValue: 0.0, required: false, description: 'Label smoothing factor', min: 0, max: 1 }
    ]
  },

  {
    id: 'mseloss',
    name: 'MSELoss',
    displayName: 'MSELoss',
    description: 'Mean Squared Error loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'l1loss',
    name: 'L1Loss',
    displayName: 'L1Loss',
    description: 'Mean Absolute Error loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'bceloss',
    name: 'BCELoss',
    displayName: 'BCELoss',
    description: 'Binary Cross Entropy loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'bcewithlogitsloss',
    name: 'BCEWithLogitsLoss',
    displayName: 'BCEWithLogitsLoss',
    description: 'BCE with built-in Sigmoid for numerical stability',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'nllloss',
    name: 'NLLLoss',
    displayName: 'NLLLoss',
    description: 'Negative Log Likelihood loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'kldivloss',
    name: 'KLDivLoss',
    displayName: 'KLDivLoss',
    description: 'Kullback-Leibler divergence loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum', 'batchmean']
      },
      { name: 'log_target', type: 'boolean', defaultValue: false, required: false, description: 'If True, target is in log-space' }
    ]
  },

  {
    id: 'smoothl1loss',
    name: 'SmoothL1Loss',
    displayName: 'SmoothL1Loss',
    description: 'Smooth L1 / Huber loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      },
      { name: 'beta', type: 'number', defaultValue: 1.0, required: false, description: 'Threshold where L1 transitions to L2 loss', min: 0 }
    ]
  },

  {
    id: 'huberloss',
    name: 'HuberLoss',
    displayName: 'HuberLoss',
    description: 'Huber loss (less sensitive to outliers)',
    icon: 'Target',
    category: 'loss',
    parameters: [
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      },
      { name: 'delta', type: 'number', defaultValue: 1.0, required: false, description: 'Threshold at which to change between L1 and L2', min: 0 }
    ]
  },

  {
    id: 'tripletmarginloss',
    name: 'TripletMarginLoss',
    displayName: 'TripletMarginLoss',
    description: 'Triplet margin loss for learning embeddings',
    icon: 'Target',
    category: 'loss',
    parameters: [
      { name: 'margin', type: 'number', defaultValue: 1.0, required: false, description: 'Has a default value of 1', min: 0 },
      { name: 'p', type: 'number', defaultValue: 2, required: false, description: 'Norm degree for pairwise distance', min: 1 },
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'cosineembeddingloss',
    name: 'CosineEmbeddingLoss',
    displayName: 'CosineEmbeddingLoss',
    description: 'Cosine embedding loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      { name: 'margin', type: 'number', defaultValue: 0.0, required: false, description: 'Should be a number from -1 to 1', min: -1, max: 1 },
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      }
    ]
  },

  {
    id: 'ctcloss',
    name: 'CTCLoss',
    displayName: 'CTCLoss',
    description: 'Connectionist Temporal Classification loss',
    icon: 'Target',
    category: 'loss',
    parameters: [
      { name: 'blank', type: 'number', defaultValue: 0, required: false, description: 'Blank label index', min: 0 },
      {
        name: 'reduction',
        type: 'select',
        defaultValue: 'mean',
        required: false,
        description: 'Reduction to apply',
        options: ['none', 'mean', 'sum']
      },
      { name: 'zero_infinity', type: 'boolean', defaultValue: false, required: false, description: 'Zero infinite losses and gradients' }
    ]
  },

  // ── Miscellaneous ────────────────────────────────────────────────────────────
  {
    id: 'transformer',
    name: 'Transformer',
    displayName: 'Transformer',
    description: 'A standard encoder-decoder Transformer model',
    icon: 'Cpu',
    category: 'misc',
    parameters: [
      { name: 'd_model', type: 'number', defaultValue: 512, required: false, description: 'Dimension of the model', min: 1 },
      { name: 'nhead', type: 'number', defaultValue: 8, required: false, description: 'Number of attention heads', min: 1 },
      { name: 'num_encoder_layers', type: 'number', defaultValue: 6, required: false, description: 'Number of encoder layers', min: 1 },
      { name: 'num_decoder_layers', type: 'number', defaultValue: 6, required: false, description: 'Number of decoder layers', min: 1 },
      { name: 'dim_feedforward', type: 'number', defaultValue: 2048, required: false, description: 'Dimension of feedforward network', min: 1 },
      { name: 'dropout', type: 'number', defaultValue: 0.1, required: false, description: 'Dropout value', min: 0, max: 1 }
    ]
  },

  {
    id: 'transformerencoderlayer',
    name: 'TransformerEncoderLayer',
    displayName: 'TransformerEncoderLayer',
    description: 'Single encoder layer of a Transformer model',
    icon: 'Cpu',
    category: 'misc',
    parameters: [
      { name: 'd_model', type: 'number', defaultValue: 512, required: true, description: 'Dimension of the model', min: 1 },
      { name: 'nhead', type: 'number', defaultValue: 8, required: true, description: 'Number of attention heads', min: 1 },
      { name: 'dim_feedforward', type: 'number', defaultValue: 2048, required: false, description: 'Dimension of feedforward network', min: 1 },
      { name: 'dropout', type: 'number', defaultValue: 0.1, required: false, description: 'Dropout value', min: 0, max: 1 },
      {
        name: 'activation',
        type: 'select',
        defaultValue: 'relu',
        required: false,
        description: 'Activation function',
        options: ['relu', 'gelu']
      },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' }
    ]
  },

  {
    id: 'transformerdecoderlayer',
    name: 'TransformerDecoderLayer',
    displayName: 'TransformerDecoderLayer',
    description: 'Single decoder layer of a Transformer model',
    icon: 'Cpu',
    category: 'misc',
    parameters: [
      { name: 'd_model', type: 'number', defaultValue: 512, required: true, description: 'Dimension of the model', min: 1 },
      { name: 'nhead', type: 'number', defaultValue: 8, required: true, description: 'Number of attention heads', min: 1 },
      { name: 'dim_feedforward', type: 'number', defaultValue: 2048, required: false, description: 'Dimension of feedforward network', min: 1 },
      { name: 'dropout', type: 'number', defaultValue: 0.1, required: false, description: 'Dropout value', min: 0, max: 1 },
      {
        name: 'activation',
        type: 'select',
        defaultValue: 'relu',
        required: false,
        description: 'Activation function',
        options: ['relu', 'gelu']
      },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' }
    ]
  },

  {
    id: 'multiheadattention',
    name: 'MultiheadAttention',
    displayName: 'MultiheadAttention',
    description: 'Multi-head attention mechanism',
    icon: 'GitBranch',
    category: 'misc',
    parameters: [
      { name: 'embed_dim', type: 'number', defaultValue: 512, required: true, description: 'Total dimension of the model', min: 1 },
      { name: 'num_heads', type: 'number', defaultValue: 8, required: true, description: 'Number of parallel attention heads', min: 1 },
      { name: 'dropout', type: 'number', defaultValue: 0.0, required: false, description: 'Dropout probability on attn_output_weights', min: 0, max: 1 },
      { name: 'bias', type: 'boolean', defaultValue: true, required: false, description: 'If True, adds bias to in/out projection layers' },
      { name: 'batch_first', type: 'boolean', defaultValue: false, required: false, description: 'If True, input is (batch, seq, feature)' }
    ]
  }
,
  {
    id: 'narrow',
    name: 'torch.narrow',
    displayName: 'Slice',
    description: 'Takes a range along one axis — the equivalent of x[..., start:start+length]. Slicing a tensor keeps the original, so the source can still feed other layers.',
    icon: 'Scissors',
    category: 'tensor',
    parameters: [
      // -1, not 1. See sliceDefaults.ts: dim 1 is the channel axis of a 4D map
      // and the feature axis of a 2D activation, so the same default meant two
      // different things. -1 is "the features" at every rank, and torch.narrow
      // takes it verbatim.
      { name: 'dim', type: 'number', defaultValue: -1, required: true, description: 'Which axis to slice. Negative counts from the right.' },
      { name: 'start', type: 'number', defaultValue: 0, required: true, description: 'First index kept', min: 0 },
      // `length`, not `end`, so the node matches torch.narrow's signature exactly
      // and codegen needs no arithmetic. length = end - start.
      // The catalogue cannot know the axis extent, so 1 is only the fallback:
      // addNode replaces it with the whole axis when the upstream shape is known,
      // which is what makes the default safe to apply unasked.
      { name: 'length', type: 'number', defaultValue: 1, required: true, description: 'How many entries to keep — that is end minus start', min: 1 }
    ]
  },
  {
    id: 'interpolate',
    // Bare, not 'F.interpolate': the backend's _LAYER_OPERATION guard allows at
    // most one prefix level so an op name cannot escape into an attribute path.
    name: 'interpolate',
    displayName: 'Interpolate',
    description: 'Resamples the spatial dimensions to a target size or by a scale factor',
    icon: 'Scaling',
    category: 'tensor',
    parameters: [
      {
        name: 'mode',
        type: 'select',
        defaultValue: 'nearest',
        required: true,
        description: 'Resampling algorithm',
        options: ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area']
      },
      // size XOR scale_factor. torch raises when given both and errors given
      // neither, so exactly one is shown at a time — see Parameter.showIf.
      { name: 'scale_factor', type: 'number', defaultValue: 2, required: false, description: 'Multiplier for the spatial size', min: 1 },
      { name: 'size', type: 'string', defaultValue: '', required: false, description: 'Explicit target size, comma-separated. Leave blank to use scale_factor.' },
      {
        name: 'align_corners',
        type: 'boolean',
        defaultValue: false,
        required: false,
        description: 'Only meaningful for the linear family; torch raises for nearest',
        showIf: { param: 'mode', oneOf: ['linear', 'bilinear', 'bicubic', 'trilinear'] }
      }
    ]
  },
  {
    id: 'add',
    name: 'torch.add',
    displayName: 'torch.add',
    description: 'PyTorch add operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sub',
    name: 'torch.sub',
    displayName: 'torch.sub',
    description: 'PyTorch sub operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'mul',
    name: 'torch.mul',
    displayName: 'torch.mul',
    description: 'PyTorch mul operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'div',
    name: 'torch.div',
    displayName: 'torch.div',
    description: 'PyTorch div operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'matmul',
    name: 'torch.matmul',
    displayName: 'torch.matmul',
    description: 'PyTorch matmul operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'bmm',
    name: 'torch.bmm',
    displayName: 'torch.bmm',
    description: 'PyTorch bmm operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'abs',
    name: 'torch.abs',
    displayName: 'torch.abs',
    description: 'PyTorch abs operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'exp',
    name: 'torch.exp',
    displayName: 'torch.exp',
    description: 'PyTorch exp operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'log',
    name: 'torch.log',
    displayName: 'torch.log',
    description: 'PyTorch log operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sqrt',
    name: 'torch.sqrt',
    displayName: 'torch.sqrt',
    description: 'PyTorch sqrt operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'pow',
    name: 'torch.pow',
    displayName: 'torch.pow',
    description: 'PyTorch pow operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sum',
    name: 'torch.sum',
    displayName: 'torch.sum',
    description: 'PyTorch sum operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'mean',
    name: 'torch.mean',
    displayName: 'torch.mean',
    description: 'PyTorch mean operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'max',
    name: 'torch.max',
    displayName: 'torch.max',
    description: 'PyTorch max operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'min',
    name: 'torch.min',
    displayName: 'torch.min',
    description: 'PyTorch min operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'argmax',
    name: 'torch.argmax',
    displayName: 'torch.argmax',
    description: 'PyTorch argmax operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'argmin',
    name: 'torch.argmin',
    displayName: 'torch.argmin',
    description: 'PyTorch argmin operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'cat',
    name: 'torch.cat',
    displayName: 'torch.cat',
    description: 'PyTorch cat operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'stack',
    name: 'torch.stack',
    displayName: 'torch.stack',
    description: 'PyTorch stack operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'chunk',
    name: 'torch.chunk',
    displayName: 'torch.chunk',
    description: 'PyTorch chunk operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'split',
    name: 'torch.split',
    displayName: 'torch.split',
    description: 'PyTorch split operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'reshape',
    name: 'torch.reshape',
    displayName: 'torch.reshape',
    description: 'PyTorch reshape operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'squeeze',
    name: 'torch.squeeze',
    displayName: 'torch.squeeze',
    description: 'PyTorch squeeze operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'unsqueeze',
    name: 'torch.unsqueeze',
    displayName: 'torch.unsqueeze',
    description: 'PyTorch unsqueeze operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'transpose',
    name: 'torch.transpose',
    displayName: 'torch.transpose',
    description: 'PyTorch transpose operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      // `dims`, not `dim`: torch, the shape engine and codegen all read `dims`, so
      // the old name was a field nothing read — the card reported no shape and
      // Export refused the node with "needs dims".
      {
        name: 'dims',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'The two dimensions to swap, e.g. 1,2'
      }
    ]
  },
  {
    id: 'permute',
    name: 'torch.permute',
    displayName: 'torch.permute',
    description: 'PyTorch permute operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      // See transpose above: the name has to be `dims`.
      {
        name: 'dims',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'The new dimension order — one entry per dimension, e.g. 0,2,3,1'
      }
    ]
  },
  {
    id: 'flatten',
    name: 'torch.flatten',
    displayName: 'torch.flatten',
    description: 'PyTorch flatten operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'clamp',
    name: 'torch.clamp',
    displayName: 'torch.clamp',
    description: 'PyTorch clamp operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'clip',
    name: 'torch.clip',
    displayName: 'torch.clip',
    description: 'PyTorch clip operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'where',
    name: 'torch.where',
    displayName: 'torch.where',
    description: 'PyTorch where operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'round',
    name: 'torch.round',
    displayName: 'torch.round',
    description: 'PyTorch round operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'floor',
    name: 'torch.floor',
    displayName: 'torch.floor',
    description: 'PyTorch floor operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'ceil',
    name: 'torch.ceil',
    displayName: 'torch.ceil',
    description: 'PyTorch ceil operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sign',
    name: 'torch.sign',
    displayName: 'torch.sign',
    description: 'PyTorch sign operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sin',
    name: 'torch.sin',
    displayName: 'torch.sin',
    description: 'PyTorch sin operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'cos',
    name: 'torch.cos',
    displayName: 'torch.cos',
    description: 'PyTorch cos operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'tan',
    name: 'torch.tan',
    displayName: 'torch.tan',
    description: 'PyTorch tan operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'asin',
    name: 'torch.asin',
    displayName: 'torch.asin',
    description: 'PyTorch asin operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'acos',
    name: 'torch.acos',
    displayName: 'torch.acos',
    description: 'PyTorch acos operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'atan',
    name: 'torch.atan',
    displayName: 'torch.atan',
    description: 'PyTorch atan operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'sinh',
    name: 'torch.sinh',
    displayName: 'torch.sinh',
    description: 'PyTorch sinh operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'cosh',
    name: 'torch.cosh',
    displayName: 'torch.cosh',
    description: 'PyTorch cosh operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },
  {
    id: 'tanh',
    name: 'torch.tanh',
    displayName: 'torch.tanh',
    description: 'PyTorch tanh operation',
    icon: 'Activity',
    category: 'tensor',
    parameters: [
      {
        name: 'dim',
        type: 'string',
        defaultValue: '',
        required: false,
        description: 'Dimension(s) to apply the operation on'
      }
    ]
  },

  // ── Code the registry does not model ─────────────────────────────────────────
  // Two placeholders, not two ops. The code parser and (from M6) the trace worker
  // emit them for code that is real but unmodelled: an opaque Python statement and
  // an instance of the user's own nn.Module. Both DECLINE a shape rather than
  // guess one — see core/op_registry.py `_shape_declines_placeholder`. They live
  // here because every id on the canvas must resolve to an OpSpec
  // (backend/tests/templates/test_palette_registry_coverage.py).
  {
    id: 'python',
    name: 'Python',
    displayName: 'Python',
    description: 'A statement from the source file that has no graph equivalent. Its code is kept verbatim and its shape is unknown until the model is traced.',
    icon: 'Hash',
    category: 'misc',
    parameters: []
  },
  {
    id: 'custom_code',
    name: 'Custom Code',
    displayName: 'Custom Code',
    description: "An instance of a user-defined nn.Module. Its structure and shapes come from the user's own file, not from this registry.",
    icon: 'Package',
    category: 'misc',
    parameters: []
  }
];

export const getLayerByType = (type: string): LayerType | undefined => {
  // Strip a canonical PyTorch module prefix ("nn.Linear", "torch.nn.Linear") the
  // same way the backend op registry does. Planner ops and LLM output routinely
  // carry the prefix even though our layer ids are bare; without this, createNode
  // silently returns undefined and the AI edit drops the node with no error.
  const normalized = type.toLowerCase().replace(/^(?:torch\.)?nn\./, '');
  return layerTypes.find(layer =>
    layer.id.toLowerCase() === normalized ||
    layer.name.toLowerCase() === normalized ||
    layer.displayName.toLowerCase() === normalized
  );
};

export const getLayersByCategory = (category: string): LayerType[] => {
  return layerTypes.filter(layer => layer.category === category);
};

// Category colors reference design tokens so light/dark mode track the
// rest of the app. Hex literals would lock the palette to one theme.
export const categories = [
  { id: 'io', name: 'Input/Output', color: 'var(--syntax-type)' },
  { id: 'module', name: 'Modules', color: 'var(--cat-module)' },
  { id: 'conv', name: 'Convolutional', color: 'var(--cat-conv)' },
  { id: 'linear', name: 'Linear', color: 'var(--cat-linear)' },
  { id: 'activation', name: 'Activation', color: 'var(--cat-activation)' },
  { id: 'pool', name: 'Pooling', color: 'var(--cat-pool)' },
  { id: 'norm', name: 'Normalization', color: 'var(--cat-norm)' },
  { id: 'recurrent', name: 'Recurrent', color: 'var(--cat-recurrent)' },
  { id: 'embedding', name: 'Embedding', color: 'var(--cat-embedding)' },
  { id: 'dropout', name: 'Dropout', color: 'var(--cat-regularization)' },
  { id: 'tensor', name: 'Tensor Ops', color: 'var(--cat-utility)' },
  { id: 'loss', name: 'Loss Functions', color: 'var(--error)' },
  { id: 'misc', name: 'Miscellaneous', color: 'var(--cat-utility)' }
];
