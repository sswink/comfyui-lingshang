from .my_nodes import SaveImageToOss
from .my_nodes import LoadMaskFromUrl
from .my_nodes import DigImageByMask
from .my_nodes import ALY_Seg_Utils
from .my_nodes import uploadOssAndGetUrl
from .my_nodes import MyGrowMaskWithBlur
from .my_nodes import ALY_Seg_Body_Utils
from .my_nodes import ALY_Seg_Common_Utils_URL
from .my_nodes import ALY_Seg_Clothes_Utils
from .my_nodes import ALY_Seg_Body_Utils_Return_crop


NODE_DISPLAY_NAME_MAPPINGS = {
    "LS_SaveImageToOss": "LS_SaveImageToOss",
    "LS_LoadMaskFromUrl":"LS_LoadMaskFromUrl",
    "LS_DigImageByMask":"LS_DigImageByMask",
    "LS_ALY_Seg_Utils":"LS_ALY_Seg_Utils",
    "LS_ALY_UploadToOssAndGetUrl":"LS_ALY_UploadToOssAndGetUrl",
    "LS_GrowMaskWithBlur":"LS_GrowMaskWithBlur",
    "LS_ALY_Seg_Body_Utils":"LS_ALY_Seg_Body_Utils",
    "LS_ALY_Seg_Common_Utils":"LS_ALY_Seg_Common_Utils",
    "LS_ALY_Seg_Clothes_Utils":"LS_ALY_Seg_Clothes_Utils",
    "LS_ALY_Seg_Body_Utils_Return_crop":"LS_ALY_Seg_Body_Utils_Return_crop"
}
NODE_CLASS_MAPPINGS = {
    "LS_SaveImageToOss": SaveImageToOss,
    "LS_LoadMaskFromUrl":LoadMaskFromUrl,
    "LS_DigImageByMask":DigImageByMask,
    "LS_ALY_Seg_Utils":ALY_Seg_Utils,
    "LS_ALY_UploadToOssAndGetUrl":uploadOssAndGetUrl,
    "LS_GrowMaskWithBlur":MyGrowMaskWithBlur,
    "LS_ALY_Seg_Body_Utils":ALY_Seg_Body_Utils,
    "LS_ALY_Seg_Common_Utils":ALY_Seg_Common_Utils_URL,
    "LS_ALY_Seg_Clothes_Utils":ALY_Seg_Clothes_Utils,
    "LS_ALY_Seg_Body_Utils_Return_crop":ALY_Seg_Body_Utils_Return_crop
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
