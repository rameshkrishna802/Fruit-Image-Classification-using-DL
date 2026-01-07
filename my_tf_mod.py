# from tensorflow.keras.models import load_model
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from PIL import Image, ImageFile
# from io import BytesIO

# quality_model=load_model('local_rotten_lr2_final.h5')
# clf_model=load_model('local_fruit_final.h5')


# # reads frfom file object
# # return array of original uploaded image and 1x100x100x3 processed image
# def preprocess(file):
#     ImageFile.LOAD_TRUNCATED_IMAGES =False
#     org_img=Image.open(BytesIO(file.read()))
#     org_img.load()
#     from PIL import Image
#     img = org_img.resize((100, 100), Image.Resampling.LANCZOS)
#     # img=org_img.resize((100,100), Image.ANTIALIAS)

#     img=image.img_to_array(img)
#     org_img=image.img_to_array(org_img)
#     return org_img, np.expand_dims(img,axis=0)


# # return [prob_for_fresh, prob_for_rotten]
# def check_rotten(img):
#     return [round(100*quality_model.predict(img)[0][0],3),round(100*(1-quality_model.predict(img)[0][0]),3)]


# # return dict... {'apple':prob, 'banana':prob, 'orange':prob}
# def classify_fruit(img):
#     fru_dict={}
#     fru_dict['apple']=round(clf_model.predict(img)[0][0]*100,4)
#     fru_dict['banana']=round(clf_model.predict(img)[0][1]*100,4)
#     fru_dict['orange']=round(clf_model.predict(img)[0][2]*100,4)

#     for value in fru_dict:
#      if fru_dict[value]<=0.001:
#         fru_dict[value]=0.00

#     return fru_dict


# 2ND CODE PART  original code

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFile
from io import BytesIO

# Load models
quality_model = load_model('local_rotten_lr2_final.h5')
clf_model = load_model('local_fruit_final.h5')

# Reads from file object
# Returns array of original uploaded image and 1x100x100x3 processed image
def preprocess(file):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    org_img = Image.open(BytesIO(file.read()))
    org_img.load()
    
    img = org_img.resize((100, 100), Image.Resampling.LANCZOS)  # ✅ Use Resampling.LANCZOS
    img = image.img_to_array(img)
    org_img = image.img_to_array(org_img)

    return org_img, np.expand_dims(img, axis=0)


# /// UPDATED CCODE
# def preprocess(file):
#     ImageFile.LOAD_TRUNCATED_IMAGES = False
#     org_img = Image.open(BytesIO(file.read()))
#     org_img.load()

#     # 🔥 Fix: Convert to RGB if RGBA or L (grayscale)
#     if org_img.mode != "RGB":
#         org_img = org_img.convert("RGB")

#     # Resize for model input
#     img = org_img.resize((100, 100), Image.Resampling.LANCZOS)

#     # Convert to array
#     img = image.img_to_array(img) / 255.0        # normalized for model
#     org_img = image.img_to_array(org_img)        # keep original (0–255) for display

#     return org_img, np.expand_dims(img, axis=0)  # (1,100,100,3)



# Return [prob_for_fresh, prob_for_rotten]
def check_rotten(img):
    pred = quality_model.predict(img)[0][0]
    return [
        round(100 * pred, 3),
        round(100 * (1 - pred), 3)
    ]

# Return dict: {'apple': prob, 'banana': prob, 'orange': prob}
def classify_fruit(img):
    preds = clf_model.predict(img)[0]
    fru_dict = {
        'apple': round(preds[0] * 100, 4),
        'banana': round(preds[1] * 100, 4),
        'orange': round(preds[2] * 100, 4)
    }

    for key in fru_dict:
        if fru_dict[key] <= 0.001:
            fru_dict[key] = 0.00

    return fru_dict



