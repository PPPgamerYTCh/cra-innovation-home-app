import numpy as np
from PIL import Image
import PIL.Image as Image
import csv
from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
from transformers import AutoFeatureExtractor, SwinForImageClassification
import warnings
from torchvision import transforms
from datasets import load_dataset
import cv2
import torch
from torch import nn
from typing import List, Callable, Optional
import os
import pandas as pd
import pydicom
import openai
from openai import OpenAI
from IPython.display import Image, display
import responses
from PIL import Image
import requests
from io import BytesIO
import io
import tensorflow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from datetime import datetime
import streamlit as st

labels = ["Normal"]
model_name_or_path = "Santipab/Esan-code-Maimeetrang-model-action-recognition"

@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def FeatureExtractor(model_name_or_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def LoadModel(model_name_or_path):
    model = SwinForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={int(i): c for i, c in enumerate(labels)},
        label2id={c: int(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True)
    return model


# Model wrapper to return a tensor
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

# """ Translate the category name to the category index.
#     Some models aren't trained on Imagenet but on even larger "data"sets,
#     so we can't just assume that 761 will always be remote-control.

# """
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
# """ Helper function to run GradCAM on an image and create a visualization.
#     (note to myself: this is probably useful enough to move into the package)
#     If several targets are passed in targets_for_gradcam,
#     e.g different categories,
#     a visualization for each of them will be created.
    
# """
    
def print_top_categories(model, img_tensor, top_k=5):
    feature_extractor = FeatureExtractor(model_name_or_path)
    inputs = feature_extractor(images=img_tensor, return_tensors="pt")
    outputs = model(**inputs)
    logits  = outputs.logits
    logits  = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    probabilities = nn.functional.softmax(logits, dim=-1)
    topK = dict()
    for i in indices:
        topK[model.config.id2label[i]] = probabilities[0][i].item()*100
    return topK

def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result



st.markdown('''
<style>
    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media (hover) {
        /* header element to be removed */
        header["data"-testid="stHeader"] {
            display: none;
        }

        /* The navigation menu specs and size */
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        /* The navigation menu open and close on hover and size */
        /* section[data-testid='stSidebar'] > div {
        height: 100%;
        width: 75px; /* Put some width to hover on. */
        /* } 

        /* ON HOVER */
        section[data-testid='stSidebar'] > div:hover{
        width: 300px;
        }

        /* The button on the streamlit navigation menu - hidden */
        button[kind="header"] {
            display: none;
        }
    }

    @media (max-width: 272px) {
        section["data"-testid='stSidebar'] > div {
            width: 15rem;
        }/.
    }
</style>
''', unsafe_allow_html=True)

# Define CSS styling for centering
centered_style = """
        display: flex;
        justify-content: center;
"""

st.markdown(
    """
<div style='border: 2px solid #00CCCC; border-radius: 5px; padding: 10px; background-color: rgba(255, 255, 255, 0.25);'>
    <h1 style='text-align: center; color: white; font-family: Arial, sans-serif; font-size: 35px;'>
    ❤️‍🩹 Diagnose and Identify Symptoms of Parkinson's Disease AI 🪫
    </h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Drawing','Action','History',], 
    iconName=['🏠','📝','🚶‍♂️','📃'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50;'>
            <h1 style='text-align: center; color: #0066CC; font-size: 100%'> แนวคิดนวัตกรรมและสิ่งประดิษฐ์  </h1>
            <h1 style='text-align: center; color: #FF8000; font-size: 100%'> ราชวิทยาลัยจุฬาภรณ์ </h1>
            <h1 style='text-align: center; color: white; font-size: 100%'> 🌎 KMUTT 💻 </h1>
        </div>
    """, unsafe_allow_html=True)

data_base = []
if tabs == 'Home':
    st.image('home.png',use_column_width=True)



