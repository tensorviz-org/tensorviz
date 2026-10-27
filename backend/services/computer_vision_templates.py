"""Computer Vision model templates and architectures"""
from typing import Dict, List, Any, Optional
from models.template_models import ModelTemplate, ModuleTemplate, PopularModel
from models.graph_models import Graph, Node, Edge, LayerType, LayerData


class ComputerVisionTemplates:
    """Service for Computer Vision architectures"""
    
    def get_all_templates(self) -> Dict[str, ModelTemplate]:
        """Get all computer vision templates"""
        return {
            "resnet": self._create_resnet_template(),
            "vgg": self._create_vgg_template(),
            "inception": self._create_inception_template(),
            "mobilenet": self._create_mobilenet_template(),
            "efficientnet": self._create_efficientnet_template(),
        }
    
    def get_module_templates(self) -> Dict[str, ModuleTemplate]:
        """Get computer vision module templates"""
        return {
            "resnet_block": self._create_resnet_block(),
            "inception_module": self._create_inception_module(),
            "depthwise_separable_conv": self._create_depthwise_separable_conv(),
            "squeeze_excitation": self._create_squeeze_excitation(),
        }
    
    def get_popular_models(self) -> Dict[str, PopularModel]:
        """Get popular computer vision models"""
        return {
            "resnet50": self._create_resnet50(),
            "vgg16": self._create_vgg16(),
            "inception_v3": self._create_inception_v3(),
            "mobilenet_v2": self._create_mobilenet_v2(),
            "efficientnet_b0": self._create_efficientnet_b0(),
        }
    
    def _create_resnet_template(self) -> ModelTemplate:
        """Create ResNet architecture template"""
        return ModelTemplate(
            id="resnet",
            name="ResNet",
            description="Residual Neural Network with skip connections for deep image classification",
            category="computer_vision",
            sub_category="cnn",
            difficulty="intermediate",
            tags=["resnet", "residual", "skip-connections", "classification"],
            
            base_config={
                "num_classes": 1000,
                "block_type": "basicblock",  # or "bottleneck"
                "layers": [3, 4, 6, 3],  # ResNet-50 configuration
                "width_per_group": 64,
                "replace_stride_with_dilation": [False, False, False],
                "norm_layer": "BatchNorm2d"
            },
            
            customizable_params=[
                {
                    "name": "num_classes",
                    "type": "number",
                    "min": 2,
                    "max": 10000,
                    "description": "Number of output classes"
                },
                {
                    "name": "block_type",
                    "type": "select",
                    "options": ["basicblock", "bottleneck"],
                    "description": "Type of residual block"
                },
                {
                    "name": "layers",
                    "type": "array",
                    "description": "Number of blocks in each layer"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "conv1",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "conv2d", "name": "Conv2d", "parameters": []},
                            "parameters": {
                                "in_channels": 3,
                                "out_channels": 64,
                                "kernel_size": 7,
                                "stride": 2,
                                "padding": 3,
                                "bias": False
                            }
                        }
                    },
                    {
                        "id": "bn1",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 100},
                        "data": {
                            "layerType": {"id": "batchnorm2d", "name": "BatchNorm2d", "parameters": []},
                            "parameters": {"num_features": 64}
                        }
                    },
                    {
                        "id": "relu",
                        "type": "layerNode",
                        "position": {"x": 500, "y": 100},
                        "data": {
                            "layerType": {"id": "relu", "name": "ReLU", "parameters": []},
                            "parameters": {"inplace": True}
                        }
                    },
                    {
                        "id": "maxpool",
                        "type": "layerNode",
                        "position": {"x": 700, "y": 100},
                        "data": {
                            "layerType": {"id": "maxpool2d", "name": "MaxPool2d", "parameters": []},
                            "parameters": {
                                "kernel_size": 3,
                                "stride": 2,
                                "padding": 1
                            }
                        }
                    },
                    {
                        "id": "layer1",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 300},
                        "data": {
                            "layerType": {"id": "resnet_layer", "name": "ResNetLayer", "parameters": []},
                            "parameters": {
                                "block": "{block_type}",
                                "planes": 64,
                                "blocks": "{layers[0]}",
                                "stride": 1
                            }
                        }
                    },
                    {
                        "id": "layer2",
                        "type": "moduleNode",
                        "position": {"x": 300, "y": 300},
                        "data": {
                            "layerType": {"id": "resnet_layer", "name": "ResNetLayer", "parameters": []},
                            "parameters": {
                                "block": "{block_type}",
                                "planes": 128,
                                "blocks": "{layers[1]}",
                                "stride": 2
                            }
                        }
                    },
                    {
                        "id": "layer3",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 300},
                        "data": {
                            "layerType": {"id": "resnet_layer", "name": "ResNetLayer", "parameters": []},
                            "parameters": {
                                "block": "{block_type}",
                                "planes": 256,
                                "blocks": "{layers[2]}",
                                "stride": 2
                            }
                        }
                    },
                    {
                        "id": "layer4",
                        "type": "moduleNode",
                        "position": {"x": 700, "y": 300},
                        "data": {
                            "layerType": {"id": "resnet_layer", "name": "ResNetLayer", "parameters": []},
                            "parameters": {
                                "block": "{block_type}",
                                "planes": 512,
                                "blocks": "{layers[3]}",
                                "stride": 2
                            }
                        }
                    },
                    {
                        "id": "avgpool",
                        "type": "layerNode",
                        "position": {"x": 400, "y": 500},
                        "data": {
                            "layerType": {"id": "adaptiveavgpool2d", "name": "AdaptiveAvgPool2d", "parameters": []},
                            "parameters": {"output_size": [1, 1]}
                        }
                    },
                    {
                        "id": "flatten",
                        "type": "layerNode",
                        "position": {"x": 600, "y": 500},
                        "data": {
                            "layerType": {"id": "flatten", "name": "Flatten", "parameters": []},
                            "parameters": {}
                        }
                    },
                    {
                        "id": "fc",
                        "type": "layerNode",
                        "position": {"x": 800, "y": 500},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": 2048,  # bottleneck expansion: 512 planes * 4
                                "out_features": "{num_classes}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "conv1", "target": "bn1"},
                    {"id": "e2", "source": "bn1", "target": "relu"},
                    {"id": "e3", "source": "relu", "target": "maxpool"},
                    {"id": "e4", "source": "maxpool", "target": "layer1"},
                    {"id": "e5", "source": "layer1", "target": "layer2"},
                    {"id": "e6", "source": "layer2", "target": "layer3"},
                    {"id": "e7", "source": "layer3", "target": "layer4"},
                    {"id": "e8", "source": "layer4", "target": "avgpool"},
                    {"id": "e9", "source": "avgpool", "target": "flatten"},
                    {"id": "e10", "source": "flatten", "target": "fc"}
                ]
            },
            
            variants=[
                {
                    "id": "resnet18",
                    "name": "ResNet-18",
                    "description": "Lightweight ResNet with 18 layers",
                    "config_overrides": {
                        "layers": [2, 2, 2, 2],
                        "block_type": "basicblock"
                    }
                },
                {
                    "id": "resnet34",
                    "name": "ResNet-34", 
                    "description": "Medium ResNet with 34 layers",
                    "config_overrides": {
                        "layers": [3, 4, 6, 3],
                        "block_type": "basicblock"
                    }
                },
                {
                    "id": "resnet50",
                    "name": "ResNet-50",
                    "description": "Deep ResNet with 50 layers using bottleneck blocks",
                    "config_overrides": {
                        "layers": [3, 4, 6, 3],
                        "block_type": "bottleneck"
                    }
                },
                {
                    "id": "resnet101",
                    "name": "ResNet-101",
                    "description": "Very deep ResNet with 101 layers",
                    "config_overrides": {
                        "layers": [3, 4, 23, 3],
                        "block_type": "bottleneck"
                    }
                }
            ],
            
            typical_use_cases=[
                "Image Classification",
                "Feature Extraction",
                "Transfer Learning",
                "Object Detection Backbone"
            ],
            
            paper_reference="Deep Residual Learning for Image Recognition (He et al., 2016)",
            minimum_input_shape=[1, 3, 32, 32],
            recommended_input_shape=[32, 3, 224, 224]
        )
    
    def _create_vgg_template(self) -> ModelTemplate:
        """Create VGG architecture template"""
        return ModelTemplate(
            id="vgg",
            name="VGG",
            description="Visual Geometry Group network with small convolution filters",
            category="computer_vision",
            sub_category="cnn",
            difficulty="beginner",
            tags=["vgg", "simple", "sequential", "classification"],
            
            base_config={
                "num_classes": 1000,
                "batch_norm": False,
                "architecture": "vgg16",  # vgg11, vgg13, vgg16, vgg19
                "dropout": 0.5
            },
            
            customizable_params=[
                {
                    "name": "architecture",
                    "type": "select", 
                    "options": ["vgg11", "vgg13", "vgg16", "vgg19"],
                    "description": "VGG variant"
                },
                {
                    "name": "batch_norm",
                    "type": "boolean",
                    "description": "Use batch normalization"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "features",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 200},
                        "data": {
                            "layerType": {"id": "vgg_features", "name": "VGGFeatures", "parameters": []},
                            "parameters": {
                                "architecture": "{architecture}",
                                "batch_norm": "{batch_norm}"
                            }
                        }
                    },
                    {
                        "id": "avgpool",
                        "type": "layerNode",
                        "position": {"x": 400, "y": 400},
                        "data": {
                            "layerType": {"id": "adaptiveavgpool2d", "name": "AdaptiveAvgPool2d", "parameters": []},
                            "parameters": {"output_size": [7, 7]}
                        }
                    },
                    {
                        "id": "classifier",
                        "type": "moduleNode",
                        "position": {"x": 840, "y": 500},
                        "data": {
                            "layerType": {"id": "vgg_classifier", "name": "VGGClassifier", "parameters": []},
                            "parameters": {
                                "num_classes": "{num_classes}",
                                "dropout": "{dropout}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "features", "target": "avgpool"},
                    {"id": "e2", "source": "avgpool", "target": "classifier"}
                ]
            },
            
            typical_use_cases=[
                "Image Classification",
                "Feature Extraction",
                "Educational Purposes"
            ],
            
            paper_reference="Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman, 2014)",
            minimum_input_shape=[1, 3, 32, 32],
            recommended_input_shape=[32, 3, 224, 224]
        )
    
    def _create_inception_template(self) -> ModelTemplate:
        """Create Inception architecture template"""
        return ModelTemplate(
            id="inception",
            name="Inception",
            description="Inception network with multi-scale convolutions and efficient computation",
            category="computer_vision",
            sub_category="cnn",
            difficulty="advanced",
            tags=["inception", "multi-scale", "efficient", "googlenet"],
            
            base_config={
                "num_classes": 1000,
                "aux_logits": True,
                "inception_blocks": [
                    {"type": "inception_a", "channels": 32},
                    {"type": "inception_b", "channels": 64},
                    {"type": "inception_c", "channels": 128}
                ]
            },
            
            customizable_params=[
                {
                    "name": "aux_logits",
                    "type": "boolean",
                    "description": "Use auxiliary classifiers"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "conv2d_1a",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "conv2d", "name": "Conv2d", "parameters": []},
                            "parameters": {
                                "in_channels": 3,
                                "out_channels": 32,
                                "kernel_size": 3,
                                "stride": 2
                            }
                        }
                    },
                    {
                        "id": "inception_blocks",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 300},
                        "data": {
                            "layerType": {"id": "inception_stack", "name": "InceptionStack", "parameters": []},
                            "parameters": {
                                "blocks": "{inception_blocks}"
                            }
                        }
                    },
                    {
                        "id": "avgpool",
                        "type": "layerNode",
                        "position": {"x": 400, "y": 500},
                        "data": {
                            "layerType": {"id": "adaptiveavgpool2d", "name": "AdaptiveAvgPool2d", "parameters": []},
                            "parameters": {"output_size": [1, 1]}
                        }
                    },
                    {
                        "id": "dropout",
                        "type": "layerNode",
                        "position": {"x": 620, "y": 500},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": 0.2}
                        }
                    },
                    {
                        "id": "flatten",
                        "type": "layerNode",
                        "position": {"x": 840, "y": 500},
                        "data": {
                            "layerType": {"id": "flatten", "name": "Flatten", "parameters": []},
                            "parameters": {}
                        }
                    },
                    {
                        "id": "fc",
                        "type": "layerNode",
                        "position": {"x": 1060, "y": 500},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": 2048,
                                "out_features": "{num_classes}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "conv2d_1a", "target": "inception_blocks"},
                    {"id": "e2", "source": "inception_blocks", "target": "avgpool"},
                    {"id": "e3", "source": "avgpool", "target": "dropout"},
                    {"id": "e4", "source": "dropout", "target": "flatten"},
                    {"id": "e5", "source": "flatten", "target": "fc"}
                ]
            },
            
            typical_use_cases=[
                "Image Classification",
                "Object Detection",
                "Feature Extraction"
            ],
            
            paper_reference="Going Deeper with Convolutions (Szegedy et al., 2015)",
            minimum_input_shape=[1, 3, 75, 75],
            recommended_input_shape=[32, 3, 299, 299]
        )
    
    def _create_mobilenet_template(self) -> ModelTemplate:
        """Create MobileNet architecture template"""
        return ModelTemplate(
            id="mobilenet",
            name="MobileNet",
            description="Efficient mobile-friendly CNN using depthwise separable convolutions",
            category="computer_vision",
            sub_category="efficient_cnn",
            difficulty="intermediate",
            tags=["mobilenet", "efficient", "mobile", "depthwise"],
            
            base_config={
                "num_classes": 1000,
                "width_mult": 1.0,
                "inverted_residual_setting": None,  # Will use default
                "round_nearest": 8,
                "block_type": "inverted_residual"
            },
            
            customizable_params=[
                {
                    "name": "width_mult",
                    "type": "select",
                    "options": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    "description": "Width multiplier for channels"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "conv1",
                        "type": "layerNode", 
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "conv2d", "name": "Conv2d", "parameters": []},
                            "parameters": {
                                "in_channels": 3,
                                "out_channels": 32,
                                "kernel_size": 3,
                                "stride": 2,
                                "padding": 1,
                                "bias": False
                            }
                        }
                    },
                    {
                        "id": "features",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 300},
                        "data": {
                            "layerType": {"id": "mobilenet_features", "name": "MobileNetFeatures", "parameters": []},
                            "parameters": {
                                "width_mult": "{width_mult}",
                                "block_type": "{block_type}"
                            }
                        }
                    },
                    {
                        "id": "avgpool",
                        "type": "layerNode",
                        "position": {"x": 400, "y": 500},
                        "data": {
                            "layerType": {"id": "adaptiveavgpool2d", "name": "AdaptiveAvgPool2d", "parameters": []},
                            "parameters": {"output_size": [1, 1]}
                        }
                    },
                    {
                        "id": "flatten",
                        "type": "layerNode",
                        "position": {"x": 620, "y": 500},
                        "data": {
                            "layerType": {"id": "flatten", "name": "Flatten", "parameters": []},
                            "parameters": {}
                        }
                    },
                    {
                        "id": "classifier",
                        "type": "layerNode",
                        "position": {"x": 400, "y": 600},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": 1280,
                                "out_features": "{num_classes}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "conv1", "target": "features"},
                    {"id": "e2", "source": "features", "target": "avgpool"},
                    {"id": "e3", "source": "avgpool", "target": "flatten"},
                    {"id": "e4", "source": "flatten", "target": "classifier"}
                ]
            },
            
            typical_use_cases=[
                "Mobile Applications",
                "Edge Computing",
                "Real-time Classification",
                "Resource-constrained Environments"
            ],
            
            paper_reference="MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard et al., 2017)",
            minimum_input_shape=[1, 3, 32, 32],
            recommended_input_shape=[32, 3, 224, 224]
        )
    
    def _create_efficientnet_template(self) -> ModelTemplate:
        """Create EfficientNet architecture template"""
        return ModelTemplate(
            id="efficientnet",
            name="EfficientNet",
            description="Systematically scaled CNN that balances depth, width, and resolution",
            category="computer_vision",
            sub_category="efficient_cnn", 
            difficulty="advanced",
            tags=["efficientnet", "scaling", "compound", "efficient"],
            
            base_config={
                "num_classes": 1000,
                "compound_coef": 0,  # B0=0, B1=1, ..., B7=7
                "width_coefficient": 1.0,
                "depth_coefficient": 1.0,
                "dropout_rate": 0.2,
                "drop_connect_rate": 0.2
            },
            
            customizable_params=[
                {
                    "name": "compound_coef",
                    "type": "select",
                    "options": [0, 1, 2, 3, 4, 5, 6, 7],
                    "description": "EfficientNet variant (B0-B7)"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "stem",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "efficientnet_stem", "name": "EfficientNetStem", "parameters": []},
                            "parameters": {
                                "out_channels": 32
                            }
                        }
                    },
                    {
                        "id": "blocks",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 300},
                        "data": {
                            "layerType": {"id": "efficientnet_blocks", "name": "EfficientNetBlocks", "parameters": []},
                            "parameters": {
                                "compound_coef": "{compound_coef}",
                                "width_coefficient": "{width_coefficient}",
                                "depth_coefficient": "{depth_coefficient}",
                                "drop_connect_rate": "{drop_connect_rate}"
                            }
                        }
                    },
                    {
                        "id": "head",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 600},
                        "data": {
                            "layerType": {"id": "efficientnet_head", "name": "EfficientNetHead", "parameters": []},
                            "parameters": {
                                "num_classes": "{num_classes}",
                                "dropout_rate": "{dropout_rate}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "stem", "target": "blocks"},
                    {"id": "e2", "source": "blocks", "target": "head"}
                ]
            },
            
            variants=[
                {
                    "id": "efficientnet_b0",
                    "name": "EfficientNet-B0",
                    "description": "Baseline EfficientNet model",
                    "config_overrides": {"compound_coef": 0}
                },
                {
                    "id": "efficientnet_b7",
                    "name": "EfficientNet-B7", 
                    "description": "Largest EfficientNet model",
                    "config_overrides": {"compound_coef": 7}
                }
            ],
            
            typical_use_cases=[
                "Image Classification",
                "Transfer Learning",
                "Mobile Applications",
                "Efficient Inference"
            ],
            
            paper_reference="EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)",
            minimum_input_shape=[1, 3, 32, 32],
            recommended_input_shape=[32, 3, 224, 224]
        )
    
    # Module template implementations
    def _create_resnet_block(self) -> ModuleTemplate:
        """Create ResNet block module template"""
        # Implementation for ResNet block
        pass
    
    def _create_inception_module(self) -> ModuleTemplate:
        """Create Inception module template"""
        # Implementation for Inception module 
        pass
    
    def _create_depthwise_separable_conv(self) -> ModuleTemplate:
        """Create depthwise separable convolution template"""
        # Implementation for depthwise separable conv
        pass
    
    def _create_squeeze_excitation(self) -> ModuleTemplate:
        """Create Squeeze-and-Excitation module template"""
        # Implementation for SE module
        pass
    
    # Popular model implementations
    def _create_resnet50(self) -> PopularModel:
        """Create ResNet-50 model configuration"""
        return PopularModel(
            id="resnet50",
            name="ResNet-50",
            full_name="50-layer Residual Network",
            description="Deep residual network with 50 layers for image classification",
            category="computer_vision",
            architecture_type="resnet",
            parameter_count="25.6M",
            training_dataset="ImageNet-1K",
            training_details={
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "batch_size": 256,
                "epochs": 90,
                "weight_decay": 1e-4,
                "momentum": 0.9
            },
            template_id="resnet",
            config={
                "layers": [3, 4, 6, 3],
                "block_type": "bottleneck",
                "num_classes": 1000
            },
            typical_applications=[
                "Image Classification",
                "Object Detection", 
                "Semantic Segmentation",
                "Feature Extraction"
            ],
            performance_metrics={
                "imagenet_top1": 76.15,
                "imagenet_top5": 92.87,
                "flops": "4.1G",
                "inference_time_ms": 12.3
            }
        )
    
    def _create_vgg16(self) -> PopularModel:
        """Create VGG-16 model configuration"""
        return PopularModel(
            id="vgg16",
            name="VGG-16",
            full_name="Visual Geometry Group 16-layer Network",
            description="16-layer VGG network with small convolution filters",
            category="computer_vision",
            architecture_type="vgg",
            parameter_count="138.4M",
            training_dataset="ImageNet-1K",
            template_id="vgg",
            config={
                "architecture": "vgg16",
                "batch_norm": False,
                "num_classes": 1000
            },
            typical_applications=[
                "Image Classification",
                "Feature Extraction",
                "Style Transfer",
                "Educational Purposes"
            ]
        )
    
    def _create_inception_v3(self) -> PopularModel:
        """Create Inception v3 model configuration"""
        return PopularModel(
            id="inception_v3",
            name="Inception v3",
            full_name="Inception v3 Deep Convolutional Architecture",
            description="Inception network optimized for computational efficiency",
            category="computer_vision",
            architecture_type="inception",
            parameter_count="23.8M",
            training_dataset="ImageNet-1K",
            template_id="inception",
            config={
                "aux_logits": True,
                "num_classes": 1000
            },
            typical_applications=[
                "Image Classification",
                "Object Detection",
                "Image Recognition",
                "Transfer Learning"
            ]
        )
    
    def _create_mobilenet_v2(self) -> PopularModel:
        """Create MobileNet v2 model configuration"""
        return PopularModel(
            id="mobilenet_v2",
            name="MobileNet v2",
            full_name="MobileNetV2: Inverted Residuals and Linear Bottlenecks",
            description="Efficient mobile network with inverted residual blocks",
            category="computer_vision",
            architecture_type="mobilenet",
            parameter_count="3.5M",
            training_dataset="ImageNet-1K",
            template_id="mobilenet",
            config={
                "width_mult": 1.0,
                "block_type": "inverted_residual",
                "num_classes": 1000
            },
            typical_applications=[
                "Mobile Applications",
                "Edge Computing",
                "Real-time Classification",
                "Object Detection"
            ]
        )
    
    def _create_efficientnet_b0(self) -> PopularModel:
        """Create EfficientNet B0 model configuration"""
        return PopularModel(
            id="efficientnet_b0",
            name="EfficientNet-B0",
            full_name="EfficientNet-B0: Baseline Model",
            description="Baseline EfficientNet with optimized scaling",
            category="computer_vision",
            architecture_type="efficientnet",
            parameter_count="5.3M",
            training_dataset="ImageNet-1K",
            template_id="efficientnet",
            config={
                "compound_coef": 0,
                "num_classes": 1000,
                "dropout_rate": 0.2
            },
            typical_applications=[
                "Image Classification",
                "Transfer Learning",
                "Mobile Applications",
                "Efficient Inference"
            ]
        )
