from PIL import Image
import PIL.Image as Image
from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")
import os
import pandas as pd
from PIL import Image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import cv2


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

with open(".../assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open(".../assets/webfonts/font.txt") as f:
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

if tabs == 'Home':
    st.image('.../home.png',use_column_width=True)
