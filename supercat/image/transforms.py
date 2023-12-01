import random
from pathlib import Path
import numpy as np
import torch
from fastai.data.transforms import image_extensions
from fastai.vision.core import PILImageBW
from fastcore.transform import DisplayedTransform
from skimage import io
from skimage.transform import resize as skresize
from torch import nn
from torchvision.transforms import v2
from skvideo.io import vreader, ffprobe
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


# class VideoResize(nn.Module):
#     def __init__(self, shape) -> None:
#         super().__init__()
#         self.shape = shape
#         # create meshgrid with batch dim
#         self.meshgrid = self._meshgrid_generator(shape).unsqueeze(0)

#     def _meshgrid_generator(self, shape):
#         d_axis = torch.linspace(-1, 1, shape[0])
#         h_axis = torch.linspace(-1, 1, shape[1])
#         w_axis = torch.linspace(-1, 1, shape[2])
#         mesh_dhw = torch.meshgrid((d_axis, h_axis, w_axis), indexing="ij")

#         return torch.stack(mesh_dhw, dim=3)

#     def resize3D(self, image):
#         # create batch dim for match grid_sample function requirement
#         # convert to float32 as required by grid_sample function requirement
#         image = image.unsqueeze(0).to(torch.float32)

#         return torch.nn.functional.grid_sample(image, self.meshgrid, align_corners=True).squeeze(0)

#     def forward(self, video_frames):
#         if len(video_frames) >= self.shape[0]:
#             frame_start = random.randint(0, len(video_frames) - self.shape[0])
#             video_frames = video_frames[frame_start : frame_start + self.shape[0]]

#         image = torch.stack(video_frames, dim=1)
#         image = self.resize3D(image)

#         return image


# class VideoCropResize(nn.Module):
#     def __init__(self, shape) -> None:
#         super().__init__()
#         self.shape = tuple(shape)
#         self.grayscale = v2.Grayscale()
        

#     def forward(self, video_frames):
#         frame_count = sum(1 for _ in video_frames)
#         video_frames.seek(0)
#         frame_start = random.randint(0, max(frame_count - self.shape[0],0))
#         frame_end = min(frame_start + self.shape[0], frame_count)
#         frame_data = []
#         for i, frame in enumerate(video_frames):
#             if i < frame_start:
#                 continue
#             if i >= frame_end:
#                 break
#             frame_data.append(self.grayscale(frame["data"]))

#         # return torch.zeros( (1,) + self.shape, dtype=torch.float16 )

#         image = np.stack(frame_data, axis=1)
#         if image.shape[2] > self.shape[1]:
#             start = random.randint(0, image.shape[2] - self.shape[1])
#             image = image[:,:,start:start+self.shape[1]]
#         if image.shape[3] > self.shape[2]:
#             start = random.randint(0, image.shape[3] - self.shape[2])
#             image = image[:,:,:,start:start+self.shape[2]]
            
#         image = image/255.0

#         if image.shape[1:] != self.shape:
#             print("resize!")
#             image = np.expand_dims(skresize(image[0], self.shape, order=3), axis=0)
        
#         return torch.tensor(image, dtype=torch.float16)


def video_shape(item:Path) -> tuple:
    metadata = ffprobe(str(item))
    frame_count = int(metadata['video']['@nb_frames'])
    frame_width = int(metadata['video']['@width'])
    frame_height = int(metadata['video']['@height'])

    return (frame_count, frame_height, frame_width)

class ImageVideoReader(DisplayedTransform):
    def __init__(self, shape) -> None:
        self.shape = list(shape)
        self.dim = len(shape)      
        self.grayscale = v2.Grayscale()


    def _rotate_info(self, shape):
        """
        Returns the shape, degree and axis for rotation process.

        If the shape is 2D, no adjustment will be applied to shape,
        and the degree and axis will be 0 and [0, 0] respectively.
        """
        if self.dim == 2:
            return shape, 0, [0, 0]

        rotate_degree = random.randint(0, 2)
        rotate_axis = random.sample([0, 1, 2], 2)

        rotate_shape = shape.copy()
        rotate_shape[rotate_axis[0]] = shape[rotate_axis[1]] if rotate_degree == 1 else rotate_shape[rotate_axis[0]]
        rotate_shape[rotate_axis[1]] = shape[rotate_axis[0]] if rotate_degree == 1 else rotate_shape[rotate_axis[1]]

        return rotate_shape, rotate_degree, rotate_axis

    def encodes(self, item: Path):
        rotate_shape, rotate_degree, rotate_axis = self._rotate_info(self.shape)
        # print('item', item)
        if item.suffix.lower() in image_extensions:
            # handle image input
            pipeline = v2.Compose([
                PILImageBW.create,
                v2.PILToTensor(),
                v2.Resize(rotate_shape[-2:], antialias=True),
                v2.ToDtype(torch.float16),
                lambda x: x / 255.0 * 2 - 1.0,
            ])

            image = pipeline(item)

            if self.dim == 3:
                image = image.unsqueeze(dim=1).expand(1, *rotate_shape)
                image = torch.rot90(image, k=rotate_degree, dims=[axis + 1 for axis in rotate_axis])
        else:
            try:
                depth, height, width = rotate_shape
                frame_count, frame_height, frame_width = video_shape(item)
                frame_start = random.randint(0, max(frame_count - depth,0))
                frame_end = min(frame_start + depth, frame_count)

                y_start = random.randint(0, max(frame_height - height,0))
                y_end = min(y_start + height, frame_height)
                x_start = random.randint(0, max(frame_width - width,0))
                x_end = min(x_start + width, frame_width)
                reader = vreader(str(item), num_frames=depth, as_grey=True)
                image = np.zeros( (frame_end-frame_start, y_end-y_start, x_end-x_start), dtype=np.uint8 )

                for i, frame in enumerate(reader):
                    if i < frame_start:
                        continue
                    image[i-frame_start,:,:] = frame[0,y_start:y_end, x_start:x_end,0]
                    
                image = image/255.0 * 2 - 1.0
                tensor = torch.tensor(image, dtype=torch.float32)
                
                if tensor.shape != tuple(rotate_shape):
                    tensor = skresize(tensor[0].detach().numpy(), rotate_shape, order=3)
                    tensor = torch.tensor(tensor, dtype=torch.float32)

                tensor = tensor.unsqueeze(0)                

                assert tensor.shape[0] == 1
                assert tensor.shape[1:] == tuple(rotate_shape)
                tensor = torch.rot90(tensor, k=rotate_degree, dims=[axis + 1 for axis in rotate_axis])
                return tensor
            except Exception as err:
                raise IOError(f"Error reading {item}:\n{err}")

        return image
    

def check_item(item, shape):
    try:
        if not item.suffix in image_extensions:
            min_size = min(video_shape(item))
            if min_size < max(shape):
                return None
        return item
    except Exception as err:
        print(f"Cannot read {item}: {err}")      
    

def check_items(items, shape):
    with joblib_progress("Checking items...", total=len(items)):
        result = Parallel(n_jobs=-1)(delayed(check_item)(item, shape) for item in items)
    
    return [item for item in result if item]
