from supercat import Supercat3d

app = Supercat3d()

for category in ["carbonate", "coal", "sandstone"]:
    for split in ["valid", "test"]:
        app.validate_individual(
            csv=f"{category}3D-{split}.csv",
            pretrained="./outputs-3d-fixed-dl3/export.pkl",
            item_dir=f"../../DeepRockSR-3D/{category}3D/{category}3D_{split}_TRI_unknown_X4/",
            do_nothing=False,
        )