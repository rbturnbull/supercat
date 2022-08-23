from supercat import Supercat

app = Supercat()

# categories = ["carbonate", "coal", "sandstone"]
# categories = ["carbonate", "coal", "sandstone", "sand"]
categories = ["sand"]
splits = ["valid", "test"]

for category in categories:
    for split in splits:
        app.validate_individual(
            csv=f"{category}2D-{split}-default_X2.csv",
            # pretrained="outputs-finetune-resnet18-sand500/export.pkl",
            pretrained="sc3d-sand-x2-default/export.pkl",
            # pretrained="outputs-finetune-resnet18-ch1-attention/export.pkl",
            item_dir=f"../../DeepRockSR-2D/{category}2D/{category}2D_{split}_BI_default_X2/",
        )