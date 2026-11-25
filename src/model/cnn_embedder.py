import torch
import torch.nn as nn
import timm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class CNNEmbedder(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        embed_dim=512,
        num_classes=None,
        dropout=0.1,
    ):
        super(CNNEmbedder, self).__init__()
        self.embed_dim = embed_dim

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Infer backbone output feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 128)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
                self.global_pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.feature_dim = features.shape[-1]
                self.global_pool = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.embedding_layer = nn.Linear(self.feature_dim, embed_dim)
        self.embedding_layer.apply(weights_init_kaiming)

        # Symmetric BNNeck head matching ViTEmbedder
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = None

    def set_freeze_stage(self, stage: str):
        """
        Progressive fine-tuning stages:
          - 'head_only': freeze entire backbone; train projection + BNNeck + classifier.
          - 'partial': unfreeze upper residual stages (layer3, layer4) + head.
          - 'full': unfreeze entire backbone + head.
        Always preserves bottleneck.bias.requires_grad == False.
        """
        if stage not in ("head_only", "partial", "full"):
            raise ValueError(f"Unknown freeze stage: {stage}")

        # Enable head parameters
        for mod in (self.embedding_layer, self.bottleneck, self.classifier):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
        self.bottleneck.bias.requires_grad_(False)

        if stage == "head_only":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif stage == "partial":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            unfrozen_any = False
            for name, child in self.backbone.named_children():
                if name in ("layer3", "layer4", "stages"):
                    if name == "stages" and isinstance(child, nn.Sequential):
                        half = len(child) // 2
                        for blk in child[half:]:
                            for p in blk.parameters():
                                p.requires_grad_(True)
                    else:
                        for p in child.parameters():
                            p.requires_grad_(True)
                    unfrozen_any = True
            if not unfrozen_any:
                # Fallback: unfreeze second half of named parameters
                params = list(self.backbone.parameters())
                for p in params[len(params) // 2 :]:
                    p.requires_grad_(True)
        elif stage == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)

        self.bottleneck.bias.requires_grad_(False)

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 0.2, include_frozen: bool = True):
        if include_frozen:
            backbone_params = list(self.backbone.parameters())
            head_params = []
            for mod in (self.embedding_layer, self.bottleneck, self.classifier):
                if mod is not None:
                    head_params.extend([p for n, p in mod.named_parameters() if not (mod is self.bottleneck and n == "bias")])
        else:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            head_params = []
            for mod in (self.embedding_layer, self.bottleneck, self.classifier):
                if mod is not None:
                    head_params.extend([p for p in mod.parameters() if p.requires_grad])

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult, "lr_scale": backbone_lr_mult})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "lr_scale": 1.0})
        return groups

    def forward(self, x, return_logits=False):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)

        raw_feat = self.embedding_layer(self.dropout(features))
        bn_feat = self.bottleneck(raw_feat)

        if return_logits and self.classifier is not None:
            cls_score = self.classifier(bn_feat)
            return raw_feat, cls_score

        return bn_feat
