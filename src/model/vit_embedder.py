import torch
import torch.nn as nn
import timm

from model.cnn_embedder import weights_init_kaiming, weights_init_classifier


class ViTEmbedder(nn.Module):
    def __init__(
        self,
        backbone_name,
        pretrained=True,
        embed_dim=512,
        num_classes=0,
        dropout=0.1,
        img_size=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )

        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            h, w = img_size if img_size is not None else (256, 128)
            with torch.no_grad():
                dummy = torch.randn(1, 3, h, w)
                out = self.backbone(dummy)
                self.in_features = out.shape[1]

        # Symmetric projection + BNNeck head matching CNNEmbedder
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_layer = nn.Linear(self.in_features, embed_dim)
        self.embedding_layer.apply(weights_init_kaiming)

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
          - 'partial': unfreeze upper half of transformer blocks + backbone norm + head.
          - 'full': unfreeze entire backbone + head.
        Always preserves bottleneck.bias.requires_grad == False.
        """
        if stage not in ("head_only", "partial", "full"):
            raise ValueError(f"Unknown freeze stage: {stage}")

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
            if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks
                half = len(blocks) // 2
                for blk in blocks[half:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
            if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
                for p in self.backbone.norm.parameters():
                    p.requires_grad_(True)
            if hasattr(self.backbone, "fc_norm") and self.backbone.fc_norm is not None:
                for p in self.backbone.fc_norm.parameters():
                    p.requires_grad_(True)
        elif stage == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)

        self.bottleneck.bias.requires_grad_(False)

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1, include_frozen: bool = True):
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
        raw_feat = self.embedding_layer(self.dropout(features))
        bn_feat = self.bottleneck(raw_feat)

        if return_logits and self.classifier is not None:
            cls_score = self.classifier(bn_feat)
            return raw_feat, cls_score

        return bn_feat
