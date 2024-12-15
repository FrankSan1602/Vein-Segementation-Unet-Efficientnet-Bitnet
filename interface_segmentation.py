import streamlit as st
import torch
from torchvision import transforms
import segmentation_models_pytorch as sm
from bitnet import replace_linears_in_pytorch_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title='Segmentación de Venas', layout = 'wide', page_icon = 'venas.png', initial_sidebar_state = 'auto')

BACKBONE = 'efficientnet-b3'
CLASSES = ['veins']
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

preprocess_input = sm.encoders.get_preprocessing_fn(BACKBONE)

# Cargar el modelo preentrenado
@st.cache_resource
def load_model():
    # Inicializa el modelo de UNet con el backbone especificado
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    
    # Reemplaza capas lineales con BitNet
    replace_linears_in_pytorch_model(model)
    
    # Carga los pesos del modelo desde el archivo
    state_dict = torch.load("best_model_bitnet.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Carga los pesos al modelo inicializado
    
    model.eval()  # Cambia el modelo a modo evaluación
    return model

# Preprocesamiento de imágenes
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Tamaño compatible con tu modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización estándar
    ])
    return preprocess(image).unsqueeze(0)

# Función para superponer la máscara predicha sobre la imagen original
# def overlay_mask(image, prediction, alpha=0.5):
#     """
#     Superpone la máscara de predicción sobre la imagen original.
    
#     Args:
#         image: Imagen original (RGB).
#         prediction: Máscara de predicción binarizada o continua.
#         alpha: Transparencia de la superposición (0.0 a 1.0).
#     Returns:
#         Imagen con la máscara superpuesta.
#     """
#     # Redimensionar la máscara de predicción al tamaño de la imagen original
#     prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

#     # Normalizar la predicción entre 0 y 1
#     normalized_prediction = (prediction_resized - prediction_resized.min()) / (
#         prediction_resized.max() - prediction_resized.min()
#     )
    
#     # Escalar la predicción a [0, 255] y binarizar las venas
#     scaled_prediction = (normalized_prediction * 255).astype(np.uint8)
#     vein_mask = scaled_prediction > 128  # Regiones correspondientes a las venas

#     # Crear una máscara semitransparente para las venas
#     color_mask = np.zeros_like(image)
#     color_mask[..., 0] = 128  # R (rojo)
#     color_mask[..., 2] = 128  # B (azul)

#     # Copiar la imagen original
#     overlay = image.copy()

#     # Aplicar el color morado solo a las venas
#     overlay[vein_mask] = (
#         (1 - alpha) * overlay[vein_mask] + alpha * color_mask[vein_mask]
#     ).astype(np.uint8)

#     return overlay

# Función para superponer la máscara predicha sobre la imagen original
def overlay_mask(image, prediction, alpha=0):
    """
    Superpone la máscara de predicción sobre la imagen original.
    
    Args:
        image: Imagen original (RGB).
        prediction: Máscara de predicción binarizada o continua.
        alpha: Transparencia de la superposición (0.0 a 1.0).
    Returns:
        Imagen con la máscara superpuesta.
    """
    # Redimensionar la máscara de predicción al tamaño de la imagen original
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Normalizar la predicción entre 0 y 1
    normalized_prediction = (prediction_resized - prediction_resized.min()) / (
        prediction_resized.max() - prediction_resized.min()
    )
    
    # Escalar la predicción a [0, 255] y convertirla a un formato binario
    scaled_prediction = (normalized_prediction * 255).astype(np.uint8)
    vein_mask = scaled_prediction > 128  # Regiones correspondientes a las venas

    # Crear una imagen completamente transparente
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Mapeo de las venas a la imagen original
    for c in range(image.shape[2]):
        overlay[vein_mask, c] = (1 - alpha) * image[vein_mask, c] + alpha * 128

    return overlay

# Realizar la predicción
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).squeeze().numpy()
    return prediction



# Interfaz gráfica con Streamlit
st.markdown("<h1 style='text-align: center; color: #3B83BD; padding: 1rem;'>- Segmentación de Venas - ESCOM IPN -</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #B61F24; padding: 1rem;'>UNET ~ EFFICIENTNET ~ BITNET</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #3B83BD; padding: 1rem;'>Franco Tadeo Sánchez García</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left; color: white;'>Carga una imagen para segmentar las venas.</h3>", unsafe_allow_html=True)

# Carga de imagen
uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Imagen Cargada", use_container_width=True)

    # Preprocesar imagen
    model = load_model()
    image_tensor = preprocess_image(image)

    # Predicción
    st.markdown("<h3 style='text-align: left; color: white;'>Generando predicción...</h3>", unsafe_allow_html=True)
    prediction = predict(model, image_tensor)

    # Mostrar resultados
    st.markdown("<h3 style='text-align: left; color: white;'>Resultados de la segmentación:</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Imagen original
    ax[0].imshow(image)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    # Imagen con la máscara superpuesta
    overlay = overlay_mask(np.array(image), prediction)
    ax[1].imshow(overlay)
    ax[1].set_title("Segmentación de Venas con UNET, EFFICIENTNET, BITNET")
    ax[1].axis("off")

    st.pyplot(fig)