
import string
import cv2
import oss2
import torch
from PIL import Image, ImageDraw, ImageFilter

from alibabacloud_imageseg20191230.models import SegmentClothAdvanceRequest, SegmentBodyRequest, \
    SegmentBodyAdvanceRequest, SegmentHDBodyAdvanceRequest, SegmentHDCommonImageRequest, \
    SegmentHDCommonImageAdvanceRequest, SegmentCommonImageAdvanceRequest, SegmentCommonImageRequest, \
    GetAsyncJobResultRequest
import os

import sys
from .AlyVision import imagese
from typing import Union, List

import json

import time
import random

import requests
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence

import numpy as np
import torchvision.transforms as transforms
from alibabacloud_tea_util.models import RuntimeOptions

import scipy.ndimage

import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))



import importlib

import folder_paths
import latent_preview
import node_helpers

# https://aishoper.oss-cn-shenzhen.aliyuncs.com/sourceImage/171276501898107rJ8S.png
class LoadMaskFromUrl:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image_url": ("STRING", {"default": ""}),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image_url, channel):
        # Send an HTTP request to the image URL
        response = requests.get(image_url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        i = Image.open(BytesIO(response.content))
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)






class DigImageByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "mask": ("MASK",),
                     },

                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "getImageByMask"

    CATEGORY = "utils"


    def getImageByMask(self, image, mask):
        image = tensor2pil(image)
        mask_image = tensor2pil(mask).convert("L")
        # 如果第二张图片的尺寸不同，我们调整它的大小
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.LANCZOS)
        transparent_background = Image.new('RGBA', image.size, (0, 0, 0, 0))
        # 把原图放到透明背景上，mask决定了透明度，mask里白色的部分会显示原图
        result = Image.composite(image, transparent_background, mask_image)
        result.save("resulted_image.png")
        return (pil2tensor(result),)







class uploadOssAndGetUrl:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "access_key": ("STRING", {"default": ""}),
                     "secret_key": ("STRING", {"default": ""}),
                     "bucket_name": ("STRING", {"default": ""}),
                     "endpoint": ("STRING", {"default": ""}),
                     "domain": ("STRING", {"default": ""}),
                     },
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "utils"


    def save_images(self, images, access_key, secret_key, bucket_name, endpoint,domain):
        characters = string.ascii_letters + string.digits
        webp_file_name = ''.join(random.choice(characters) for i in range(10))
        webp_file_name = 'segment/'+webp_file_name+".webp"
        # ...转换图片数据和构建元数据的逻辑保持不变...
        image = tensor2pil(images)
        # 清除图片元数据
        data = list(image.getdata())
        clean_image = Image.new(image.mode, image.size)
        clean_image.putdata(data)
        # 重置文件指针到开始位置
        auth = oss2.Auth(access_key, secret_key)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        # 直接上传到OSS
        upload_webp_oss(clean_image, webp_file_name,bucket)

        webp_url = domain+"/"+webp_file_name

        return {"ui": {"text": (webp_url,)}, "result": (webp_url,)}



class SaveImageToOss:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "png_file_name": ("STRING", {"default": ""}),
                     "webp_file_name": ("STRING", {"default": ""}),
                     "access_key": ("STRING", {"default": ""}),
                     "secret_key": ("STRING", {"default": ""}),
                     "bucket_name": ("STRING", {"default": ""}),
                     "endpoint": ("STRING", {"default": ""}),
                     "domain": ("STRING", {"default": ""}),
                     },

                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"


    def save_images(self, images, png_file_name,webp_file_name, access_key, secret_key, bucket_name, endpoint,domain):
        # ...转换图片数据和构建元数据的逻辑保持不变...

        image = tensor2pil(images)
        # 清除图片元数据
        data = list(image.getdata())
        clean_image = Image.new(image.mode, image.size)
        clean_image.putdata(data)
        # 重置文件指针到开始位置
        auth = oss2.Auth(access_key, secret_key)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        # 直接上传到OSS
        upload_webp_oss(clean_image, webp_file_name,bucket)
        upload_png_oss(clean_image, png_file_name,bucket)

        png_url = domain+"/"+png_file_name
        webp_url = domain+"/"+webp_file_name

        return {"ui": {"text": [png_url,webp_url]}, "result": (png_url,webp_url)}

class ALY_Seg_Common_Utils_URL:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_url":("STRING", {"default": ""}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "processor"
    CATEGORY = "utils"

    def processor(self,image_url,access_key,secret_key):
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        #
        binary_io = BytesIO()
        image.save(binary_io, format='WEBP', quality=100)
        binary_io.seek(0)
        width, height = image.size
        if width >= 2000 or height >= 2000:
            segment_common_request = SegmentHDCommonImageAdvanceRequest()
            segment_common_request.image_url_object =binary_io

        else:
            segment_common_request = SegmentCommonImageAdvanceRequest()
            segment_common_request.image_url_object =binary_io



        runtime = RuntimeOptions()
        result_image_url = ''

        try:
            # 初始化Client
            client = imagese.create_client_json(access_key,secret_key)
            if width >= 2000 or height >= 2000:
                response = client.segment_hdcommon_image_advance(segment_common_request,runtime)

                jobid = response.body.request_id
                time.sleep(5)
                job_request = GetAsyncJobResultRequest(jobid)
                response = client.get_async_job_result(job_request)

                result = response.body.data.result
                result_image_url = json.loads(result)["imageUrl"]
            else:
                response = client.segment_common_image_advance(segment_common_request,runtime)
                result_image_url = response.body.data.image_url





        except Exception as error:
            # 获取整体报错信息
            print("==========错误 start===========")
            print(error)
            print("==========错误 end===========")

        # source_img = get_image_from_url(result_image_url)
        # combined_mask = add_masks(pil2tensor(source_img),pil2tensor(img_from_url(class_urls[0])),pil2tensor(img_from_url(class_urls[1])),pil2tensor(img_from_url(class_urls[2])))

        return {"ui": {"text": (result_image_url,)}, "result": (result_image_url,)}

class ALY_Seg_Body_Utils_Return_crop:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image":("IMAGE", {"default": "","multiline": False}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("IMAGE","MASK")

    FUNCTION = "processor"
    CATEGORY = "utils"

    def processor(self,image,access_key,secret_key):
        image = tensor2pil(image)

        binary_io = BytesIO()
        image.save(binary_io, format='WEBP', quality=100)
        binary_io.seek(0)
        width, height = image.size
        if width >= 2000 or height >= 2000:
            segment_body_request = SegmentHDBodyAdvanceRequest()
        else:
            segment_body_request = SegmentBodyAdvanceRequest()

        # segment_cloth_request = SegmentClothAdvanceRequest()
        segment_body_request.image_urlobject =binary_io
        # classes = ['shoes','bag','hat']
        # segment_cloth_request.cloth_class = classes
        #
        # segment_body_request.return_form = 'crop'

        runtime = RuntimeOptions()
        image_url = ''

        try:
            # 初始化Client
            client = imagese.create_client_json(access_key,secret_key)
            if width >= 2000 or height >= 2000:
                response = client.segment_hdbody_advance(segment_body_request, runtime)
            else:
                response = client.segment_body_advance(segment_body_request, runtime)

            image_url = response.body.data.image_url



        except Exception as error:
            # 获取整体报错信息
            print(error)


        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        output_images = []
        output_masks = []

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class ALY_Seg_Body_Utils:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image":("IMAGE", {"default": "","multiline": False}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "processor"
    CATEGORY = "utils"

    def processor(self,image,access_key,secret_key):
        image = tensor2pil(image)

        binary_io = BytesIO()
        image.save(binary_io, format='WEBP', quality=100)
        binary_io.seek(0)
        width, height = image.size
        if width >= 2000 or height >= 2000:
            segment_body_request = SegmentHDBodyAdvanceRequest()
        else:
            segment_body_request = SegmentBodyAdvanceRequest()

        # segment_cloth_request = SegmentClothAdvanceRequest()
        segment_body_request.image_urlobject =binary_io
        # classes = ['shoes','bag','hat']
        # segment_cloth_request.cloth_class = classes
        #
        segment_body_request.return_form = 'mask'

        runtime = RuntimeOptions()
        image_url = ''

        try:
            # 初始化Client
            client = imagese.create_client_json(access_key,secret_key)
            if width >= 2000 or height >= 2000:
                response = client.segment_hdbody_advance(segment_body_request, runtime)
            else:
                response = client.segment_body_advance(segment_body_request, runtime)

            image_url = response.body.data.image_url



        except Exception as error:
            # 获取整体报错信息
            print(error)


        source_img = get_image_from_url(image_url)
        # combined_mask = add_masks(pil2tensor(source_img),pil2tensor(img_from_url(class_urls[0])),pil2tensor(img_from_url(class_urls[1])),pil2tensor(img_from_url(class_urls[2])))

        return (pil2tensor(source_img),)


class ALY_Seg_Clothes_Utils:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image":("IMAGE", {"default": "","multiline": False}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "processor"
    CATEGORY = "utils"

    def processor(self,image,access_key,secret_key):
        image = tensor2pil(image)

        binary_io = BytesIO()
        image.save(binary_io, format='WEBP', quality=100)
        binary_io.seek(0)
        segment_cloth_request = SegmentClothAdvanceRequest()
        segment_cloth_request.image_urlobject =binary_io
        # classes = ['shoes','bag','hat']
        # segment_cloth_request.cloth_class = classes

        segment_cloth_request.return_form = 'mask'

        runtime = RuntimeOptions()
        image_url = ''
        class_urls =[]
        try:
            # 初始化Client
            client = imagese.create_client_json(access_key,secret_key)
            response = client.segment_cloth_advance(segment_cloth_request, runtime)
            image_url = response.body.data.elements[0].image_url
            # class_url = response.body.data.elements[1].class_url
            #
            # for clothe_class in classes:
            #     class_urls.append(class_url[clothe_class])
        except Exception as error:
            # 获取整体报错信息
            print("==========错误 start===========")
            print(error)
            print("==========错误 end===========")

        source_img = img_from_url(image_url)
        # combined_mask = add_masks(pil2tensor(source_img),pil2tensor(img_from_url(class_urls[0])),pil2tensor(img_from_url(class_urls[1])),pil2tensor(img_from_url(class_urls[2])))

        return (pil2tensor(source_img))

class ALY_Seg_Utils:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image":("IMAGE", {"default": "","multiline": False}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "processor"
    CATEGORY = "utils"

    def processor(self,image,access_key,secret_key):
        image = tensor2pil(image)

        binary_io = BytesIO()
        image.save(binary_io, format='WEBP', quality=100)
        binary_io.seek(0)
        # 获取字节数据
        image_bytes = binary_io.getvalue()
        # 打印字节长度

        segment_cloth_request = SegmentClothAdvanceRequest()
        segment_cloth_request.image_urlobject =binary_io
        classes = ['shoes','bag','hat']
        segment_cloth_request.cloth_class = classes

        segment_cloth_request.return_form = 'mask'

        runtime = RuntimeOptions()
        image_url = ''
        class_urls =[]
        try:
            # 初始化Client
            client = imagese.create_client_json(access_key,secret_key)
            response = client.segment_cloth_advance(segment_cloth_request, runtime)
            image_url = response.body.data.elements[0].image_url
            class_url = response.body.data.elements[1].class_url

            for clothe_class in classes:
                class_urls.append(class_url[clothe_class])
        except Exception as error:
            # 获取整体报错信息
            print("==========错误 start===========")
            print(error)
            print("==========错误 end===========")

        source_img = img_from_url(image_url)
        combined_mask = add_masks(pil2tensor(source_img),pil2tensor(img_from_url(class_urls[0])),pil2tensor(img_from_url(class_urls[1])),pil2tensor(img_from_url(class_urls[2])))

        return (combined_mask)


def add_masks(mask1, mask2, mask3, mask4):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    mask3 = mask3.cpu()
    mask4 = mask4.cpu()

    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255
    cv2_mask3 = np.array(mask3) * 255
    cv2_mask4 = np.array(mask4) * 255

    if cv2_mask1.shape == cv2_mask2.shape == cv2_mask3.shape == cv2_mask4.shape:
        mask_temp2 = cv2.add(cv2_mask1, cv2_mask2)
        mask_temp3 = cv2.add(mask_temp2, cv2_mask3)
        mask_temp4 = cv2.add(mask_temp3, cv2_mask4)
        return torch.clamp(torch.from_numpy(mask_temp4) / 255.0, min=0, max=1)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1

def upload_webp_oss(image, oss_path,bucket):
    """将图片对象直接上传到阿里云OSS。"""
    # 创建一个BytesIO对象
    image_stream = BytesIO()
    # 保存图片到BytesIO对象
    image.save(image_stream, format='WEBP', quality=100)
    image_stream.seek(0)
    # 上传到OSS
    bucket.put_object(oss_path, image_stream)

def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        image = Image.open(BytesIO(response.content))
        return image
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

#转灰度图 黑白蒙版
def img_from_url(imageUrl):
    response = requests.get(imageUrl)
    # 将内容转换为字节流
    image_bytes = BytesIO(response.content)
    # 用Pillow打开这个字节流，将其转换为一个图像对象
    image = Image.open(image_bytes)
    # 如果需要，可以将其转换为灰度图像
    image = image.convert('L')
    return image



def upload_png_oss(image, oss_path,bucket,metadata=None):
    """将图片对象直接上传到阿里云OSS。"""
    # 创建一个BytesIO对象
    image_stream = BytesIO()
    # 保存图片到BytesIO对象
    image.save(image_stream, format='PNG', compress_level=0, pnginfo=metadata)
    image_stream.seek(0)
    # 上传到OSS
    bucket.put_object(oss_path, image_stream)

#纯图片转tensor
def pil_to_tensor(pil_image):
    transform = transforms.ToTensor()
    tensor_image = transform(pil_image)
    return tensor_image

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Utility functions from mtb nodes: https://github.com/melMass/comfy_mtb
def kjPil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def kjTensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def check_image_mask(image, mask, name):
    if len(image.shape) < 4:
        # image tensor shape should be [B, H, W, C], but batch somehow is missing
        image = image[None,:,:,:]

    if len(mask.shape) > 3:
        # mask tensor shape should be [B, H, W] but we get [B, H, W, C], image may be?
        # take first mask, red channel
        mask = (mask[:,:,:,0])[:,:,:]
    elif len(mask.shape) < 3:
        # mask tensor shape should be [B, H, W] but batch somehow is missing
        mask = mask[None,:,:]

    if image.shape[0] > mask.shape[0]:

        if mask.shape[0] == 1:

            mask = torch.cat([mask] * image.shape[0], dim=0)
        else:

            empty_mask = torch.zeros([image.shape[0] - mask.shape[0], mask.shape[1], mask.shape[2]])
            mask = torch.cat([mask, empty_mask], dim=0)
    elif image.shape[0] < mask.shape[0]:

        mask = mask[:image.shape[0],:,:]

    return (image, mask)

MAX_RESOLUTION=16384


class MyGrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("FLOAT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000,
                    "step": 0.1
                }),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "utils"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
# GrowMaskWithBlur
- mask: Input mask or mask batch
- expand: Expand or contract mask or mask batch by a given amount
- incremental_expandrate: increase expand rate by a given amount per frame
- tapered_corners: use tapered corners
- flip_input: flip input mask
- blur_radius: value higher than 0 will blur the mask
- lerp_alpha: alpha value for interpolation between frames
- decay_factor: decay value for interpolation between frames
- fill_holes: fill holes in the mask (slow)"""

    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        width = mask.shape[-2]
        height = mask.shape[-1]
        if width > height:
            resolution = width
        else:
            resolution = height
        expand = expand * resolution/10
        blur_radius = blur_radius * resolution/10

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy()
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = kjTensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = kjPil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)


