import streamlit as st
import cv2
import numpy as np
import os

# --- KÜTÜPHANE YÜKLEME ---
try:
    import tensorflow as tf
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        from keras.models import load_model
except ImportError:
    st.error("HATA: TensorFlow bulunamadı.")
    st.stop()
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Hesap Makinesi", layout="wide")
st.title("🧮 Yapay Zeka Hesap Makinesi (Sade)")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_my_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # YENİ MODEL DOSYA ADI
    model_path = os.path.join(current_dir, 'hesap_makinesi_model_v4_no_xyz.h5') 
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model dosyası bulunamadı! '{model_path}' dosyası eksik.")
        return None
    return load_model(model_path)

model = load_my_model()

# --- GÜNCELLENMİŞ ETİKETLER (X, Y, Z YOK) ---
# Alfabetik sıraya göre: 0-9, add, dec, div, eq, mul, sub
# Colab çıktısını yine de kontrol etmekte fayda var ama %99 böyle olacaktır:
labels = { 
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
    10: 'add', 11: 'dec', 12: 'div', 13: 'eq', 14: 'mul', 15: 'sub'
}

map_symbols = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'eq': '=', 'dec': '.'
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Çizim Alanı")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)", 
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=500,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Temizle"):
        st.rerun()

with col2:
    st.subheader("2. Kontrol ve Sonuç")
    
    ters_cevir = st.checkbox("Renkleri Ters Çevir (Hata alırsan dene)", value=False)
    
    if st.button('Hesapla', type="primary"):
        if model and canvas_result.image_data is not None:
            
            # Görüntü İşleme
            img_data = canvas_result.image_data.astype('uint8')
            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                bounding_boxes = [cv2.boundingRect(c) for c in contours]
                (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

                equation_str = ""
                debug_images = []

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h < 50: continue
                    
                    padding = 15
                    roi = gray[max(0, y-padding):min(gray.shape[0], y+h+padding), 
                               max(0, x-padding):min(gray.shape[1], x+w+padding)]
                    
                    if ters_cevir:
                        roi = cv2.bitwise_not(roi)
                    
                    try:
                        roi = cv2.resize(roi, (28, 28))
                    except:
                        continue
                        
                    debug_images.append(roi)

                    roi_norm = roi / 255.0
                    roi_norm = np.expand_dims(roi_norm, axis=0)
                    roi_norm = np.expand_dims(roi_norm, axis=-1)

                    prediction = model.predict(roi_norm, verbose=0)
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    label = labels.get(class_index, "?")
                    symbol = map_symbols.get(label, label)
                    
                    equation_str += symbol
                    
                    cv2.rectangle(img_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_data, symbol, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                st.image(img_data, caption="Algılanan", width=400)
                st.write("🔍 Modelin Gözünden:")
                st.image(debug_images, width=60)
                
                st.info(f"Denklem: {equation_str}")
                
                # --- HESAPLAMA (Artık değişken kontrolü yok, dümdüz hesap) ---
                try:
                    clean_eq = equation_str.replace("=", "")
                    # Sadece izin verilen karakterler kalsın (Güvenlik)
                    allowed = set("0123456789+-*/.")
                    if set(clean_eq).issubset(allowed):
                         st.success(f"## Sonuç: {eval(clean_eq)}")
                    else:
                        st.warning("Tanınmayan karakterler var, hesaplanamadı.")
                except:
                    st.error("Matematiksel hata (Örn: 0'a bölme veya eksik işlem).")
            else:
                st.warning("Lütfen bir işlem çizin.")