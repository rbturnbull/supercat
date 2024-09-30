from pathlib import Path
from rich.console import Console
from rich.progress import track
import numpy as np
import torchapp as ta
import pandas as pd

from .metrics import smooth_l1_loss, psnr
from .models import ResidualUNet, calc_initial_features_residualunet
from .enums import PaddingMode
# from .diffusion import DDPMCallback, DDPMSamplerCallback
from .data import SupercatDataModule, TrainingItem

console = Console()

class Supercat(ta.TorchApp):
    @ta.method
    def setup(
        self,
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
    ):
        self.dim = dim
        self.in_channels = 1

    @ta.method
    def data(
        self,
        csv:Path = ta.Param(help="The path to the CSV file."),
        base_dir: Path = ta.Param(
            default=None, 
            help="The base directory for images with relative paths. "
                "If not given, then it is relative to the csv directory."
        ),
        batch_size:int = ta.Param(default=10, help="The batch size."),
        scale_factor:float = ta.Param(default=2.0, help="The factor to upscale the image."),
        validation_partition:int = ta.Param(default=0, help="The partition of the data to use for validation."),
        validation_proportion:float = ta.Param(default=0.2, help="The proportion of the data to use for validation if not specified in the CSV."),
        max_samples:int = ta.Param(default=None, help="If set, then the number of input samples for training/validation is truncated at this number."),
        num_workers:int = ta.Param(default=4, help="The number of workers to use for loading data."),
        width:int=0,
        height:int=0,
        depth:int=0,
    ) -> SupercatDataModule:
        """
        Creates a data module which Supercat uses in training and prediction.
        """
        csv = Path(csv)
        base_dir = base_dir or Path(csv).parent
        base_dir = Path(base_dir)

        df = pd.read_csv(csv)
        if 'validation' not in df:
            if 'partition' in df:
                df['validation'] = df['partition'] == validation_partition
            else:
                # assign randomly if no partition column
                df['validation'] = np.random.rand(len(df)) < validation_proportion

        training_data = []
        validation_data = []
        for _, row in df.iterrows():
            high_res = base_dir/row['high_res']
            upscaled = base_dir/row['upscaled'] if 'upscaled' in row else None

            item = TrainingItem(high_res, upscaled)

            dataset = validation_data if row['validation'] else training_data
            if max_samples and len(dataset) > max_samples:
                continue
            dataset.append( item )


        return SupercatDataModule(
            training_items=training_data,
            validation_items=validation_data,
            scale_factor=scale_factor,
            batch_size=batch_size,
            num_workers=num_workers,
            width=width,
            height=height,
            depth=depth,
        )
    
    @ta.method
    def metrics(self, **kwargs):
        return [
            ("psnr", psnr),
        ]

    @ta.method
    def model(
        self, 
        pretrained:Path=None,
        initial_features:int = ta.Param(
            None,
            help="The number of features after the initial CNN layer. If not set then it is derived from the MACC."
        ),
        growth_factor:float = ta.Param(
            2.0,
            tune=True, 
            tune_min=1.0,
            tune_max=4.0,
            tune_log=True,
            help="The factor to grow the number of convolutional filters each time the model downscales."
        ),
        kernel_size:int = ta.Param(
            3,
            tune=True, 
            tune_choices=[3,5,7],
            help="The size of the kernel in the convolutional layers."
        ),
        stub_kernel_size:int = ta.Param(
            7,
            tune=True, 
            tune_choices=[5,7,9],
            help="The size of the kernel in the initial stub convolutional layer."
        ),
        downblock_layers:int = ta.Param(
            4,
            tune=True, 
            tune_min=2,
            tune_max=5,
            help="The number of layers to downscale (and upscale) in the UNet."
        ),
        attn_layers:str = ta.Param(
            "",
            help="Whether or not to use self attention in the model. Specify the indices of the layers, seperated with ',', to include self attention layer. Index starts from 0."
        ),
        position_emb_dim:int = ta.Param(
            None,
            help="The dimension of the positional embedding. If not set, the model will not be conditioned on positional info."
        ),
        affine:bool = ta.Param(
            False,
            help="Whether or not to use affine transformations in feature wise transformation."
        ),
        macc:int = ta.Param(
            default=132_000,
            help=(
                "The approximate number of multiply or accumulate operations in the model per pixel/voxel. " +
                "Used to set initial_features if it is not provided explicitly."
            ),
        ),
        padding_mode: PaddingMode = ta.Param(
            PaddingMode.REFLECT.value, 
            help="The padding mode for convolution layers", 
            case_sensitive=False
        ),
        **kwargs,
    ):
        if pretrained:
            module_class = self.module_class(**kwargs)
            module = module_class.load_from_checkpoint(pretrained)
            return module.model

        dim  = self.dim
        attn_layers = tuple(map(int, filter(None, attn_layers.split(','))))

        padding_mode = str(padding_mode)

        if not initial_features:
            assert macc

            initial_features = calc_initial_features_residualunet(
                macc=macc,
                dim=dim,
                growth_factor=growth_factor,
                kernel_size=kernel_size,
                stub_kernel_size=stub_kernel_size,
                downblock_layers=downblock_layers,
            )

        return ResidualUNet(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.in_channels,
            out_channels=1,
            initial_features=initial_features,
            growth_factor=growth_factor,
            kernel_size=kernel_size,
            downblock_layers=downblock_layers,
            attn_layers=attn_layers,
            position_emb_dim=position_emb_dim,
            use_affine=affine,
        )

    @ta.method
    def loss_function(self):
        """
        Returns the loss function to use with the model.
        """
        return smooth_l1_loss

    # def inference_dataloader(
    #     self, 
    #     learner, 
    #     dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
    #     items:List[Path] = None, 
    #     item_dir: Path = ta.Param(None, help="A directory with images to upscale."), 
    #     width:int = ta.Param(500, help="The width of the final image/volume."), 
    #     height:int = ta.Param(None, help="The height of the final image/volume."), 
    #     depth:int = ta.Param(None, help="The depth of the final image/volume."), 
    #     start_x:int=None,
    #     end_x:int=None,
    #     start_y:int=None,
    #     end_y:int=None,
    #     start_z:int=None,
    #     end_z:int=None,        
    #     **kwargs
    # ):  
    #     self.dim = dim

    #     if not items:
    #         items = []
    #     if isinstance(items, (Path, str)):
    #         items = [items]
    #     if item_dir:
    #         items += self.get_items(item_dir)

    #     items = [Path(item) for item in items]
    #     self.items = items
    #     dataloader = learner.dls.test_dl(items, with_labels=True, **kwargs)
    #     dataloader.transform = dataloader.transform[:1] # ignore the get_y function
    #     height = height or width
    #     depth = depth or width
        
    #     interpolation = InterpolateTransform(depth=depth, height=height, width=width, dim=dim)
    #     crop_transform = CropTransform(
    #         start_x=start_x, end_x=end_x,
    #         start_y=start_y, end_y=end_y,
    #         start_z=start_z, end_z=end_z,
    #     )
    #     self.rescaling = RescaleImageMinMax()
    #     dataloader.after_item = Pipeline( [crop_transform, interpolation, self.rescaling, ToTensor] )
    #     if isinstance(dataloader.after_batch[1], RescaleImage):
    #         dataloader.after_batch = Pipeline( *(dataloader.after_batch[:1] + dataloader.after_batch[2:]) ) if dim == 2 else Pipeline([])

    #     return dataloader

    # def output_results(
    #     self, 
    #     results, 
    #     return_data:bool=False, 
    #     output_dir: Path = ta.Param(None, help="The location of the output directory. If not given then it uses the directory of the item."),
    #     suffix:str = ta.Param("", help="The file extension for the output file."),
    #     **kwargs,
    # ):
    #     list_to_return = []
    #     if output_dir:
    #         output_dir = Path(output_dir)
    #         output_dir.mkdir(exist_ok=True, parents=True)

    #     for item, result in zip(self.items, results[0]):
    #         my_suffix = suffix or item.suffix
    #         if my_suffix[0] != ".":
    #             my_suffix = "." + my_suffix

    #         new_name = item.with_suffix("").name + f".upscaled{my_suffix}"
    #         my_output_dir = output_dir or item.parent
    #         new_path = my_output_dir/new_name

    #         dim = len(result.shape) - 1
    #         if dim == 2:
    #             # hack get extrema to rescale
    #             data = np.asarray(Image.open(item).convert('L'))
    #             min, max = Image.open(item).convert('L').getextrema()
    #             result[0] = self.rescaling.decodes(result[0], min, max)

    #             pixels = torch.clip(result[0], min=0, max=255)
    #             im = Image.fromarray( pixels.cpu().detach().numpy().astype('uint8') )
    #             im.save(new_path)
    #         else:
    #             # hack get extrema to rescale
    #             data = read3D(item)
    #             min, max = data.min(), data.max()
    #             result[0] = self.rescaling.decodes(result[0], min, max)

    #             write3D(new_path, result[0].cpu().detach().numpy())            
                            
    #         list_to_return.append(result[0] if return_data else new_path)
    #         console.print(f"Upscaled '{item}' ⮕ '{new_path}'")

    #     return list_to_return
    
    # def pretrained_location(
    #     self,
    #     dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
    # ) -> str:
    #     assert dim in [2,3]
    #     if dim == 2:
    #         return f"https://github.com/rbturnbull/supercat/releases/download/v0.2.1/supercat-{dim}D.0.2.pkl"
    #     return f"https://github.com/rbturnbull/supercat/releases/download/v0.3.0/supercat-{dim}D.0.3.pkl"        


# class SupercatDiffusion(Supercat):
#     in_channels = 2
    
#     def extra_callbacks(self):
#         return [DDPMCallback()]
    
#     def inference_callbacks(self):
#         return [DDPMSamplerCallback()]        

#     def pretrained_location(
#         self,
#         dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
#     ) -> str:
#         assert dim in [2,3]
#         if dim == 2:
#             return f"https://github.com/rbturnbull/supercat/releases/download/v0.2.1/supercat-diffusion-{dim}D.0.2.pkl"
#         return f"https://github.com/rbturnbull/supercat/releases/download/v0.3.0/supercat-diffusion-{dim}D.0.3.pkl"

    # def output_results(
    #     self, 
    #     results, 
    #     output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
    #     diffusion_gif:bool=False,        
    #     diffusion_gif_fps:float=ta.Param(120.0, help="The frames per second to use when generating the gif."),
    #     **kwargs,
    # ):
    #     breakpoint()
    #     # final_results = [[result[-1] for result in results[0][0]]]
    #     to_return = super().output_results(results, output_dir=output_dir, **kwargs)

    #     if diffusion_gif:
    #         assert self.dim == 2

    #         output_dir = Path(output_dir)
    #         print(f"Saving {len(results[0])} generated images:")

    #         transform = T.ToPILImage()
    #         output_dir.mkdir(exist_ok=True, parents=True)
    #         images = []
    #         for index, image in enumerate(results[0][0]):
    #             path = output_dir/f"image.{index}.png"
                
    #             image = transform(torch.clip(image[0]/2.0 + 0.5, min=0.0, max=1.0))
    #             images.append(image)
    #         print(f"\t{path}")
    #         images[0].save(output_dir/f"image.gif", save_all=True, append_images=images[1:], fps=diffusion_gif_fps)

    #     return to_return

