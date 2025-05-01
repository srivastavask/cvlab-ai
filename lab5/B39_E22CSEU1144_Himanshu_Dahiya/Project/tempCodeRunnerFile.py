# First install required packages
from tensorflow.keras.utils import pad_sequences

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Missing import
import pickle