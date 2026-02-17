import streamlit as st
import cv2
import numpy as np
import joblib
import time
from skimage.feature import local_binary_pattern

# CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Nexus-Eye | Forensic Terminal", layout="wide")

# STYLE CSS AVANCÉ (GHOST DESIGN & NEON)
st.markdown("""
    <style>
    /* Fond global */
    .stApp { background: #000408; color: #00f2fe; font-family: 'Segoe UI', Roboto, Helvetica; }
    
    /* Barre latérale (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #000a12;
        border-right: 1px solid #00f2fe;
        box-shadow: 5px 0 15px rgba(0, 242, 254, 0.1);
    }
    
    /* Titre Principal */
    .main-header {
        font-size: 55px; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #00f2fe, #005f73);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px; letter-spacing: -1px;
    }
    .sub-header { text-align: center; font-size: 14px; letter-spacing: 5px; color: #005f73; margin-bottom: 40px; }

    /* Cartes de métriques */
    .metric-box {
        background: rgba(0, 242, 254, 0.02);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 12px;
        padding: 20px;
        transition: 0.3s;
    }
    .metric-box:hover { border-color: #00f2fe; background: rgba(0, 242, 254, 0.05); }
    .m-val { font-size: 32px; font-weight: bold; color: #fff; display: block; }
    .m-label { font-size: 12px; color: #00f2fe; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
    .m-desc { font-size: 11px; color: #666; margin-top: 8px; line-height: 1.3; }

    /* Zone d'affichage image */
    .img-container { border: 1px solid #1a1a1a; border-radius: 15px; padding: 10px; background: #000; }

    /* Bouton Diagnostic */
    .stButton>button {
        width: 100%; border-radius: 8px; height: 60px;
        background: transparent; color: #00f2fe; border: 2px solid #00f2fe;
        font-weight: bold; font-size: 18px; letter-spacing: 2px;
        transition: 0.4s; margin-top: 20px;
    }
    .stButton>button:hover { background: #00f2fe; color: #000; box-shadow: 0 0 30px rgba(0, 242, 254, 0.4); }

    /* Verdict */
    .verdict-box {
        margin-top: 25px; padding: 30px; border-radius: 15px; 
        text-align: center; border: 1px solid #00f2fe;
        background: rgba(0, 242, 254, 0.02);
    }
    </style>
    """, unsafe_allow_html=True)

# LOGIQUE TECHNIQUE (STABLE)
def get_analysis(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
    img_res = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # LBP
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float") / (hist.sum() + 1e-7)

    # Gabor & Canny
    kernel = cv2.getGaborKernel((31, 31), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    f_gabor = cv2.filter2D(gray, cv2.CV_8U, kernel)
    canny = cv2.Canny(gray, 80, 180) # Ajusté pour plus de finesse

    # Caractéristiques pour le modèle
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3)))
    m_val, s_val = cv2.meanStdDev(gray)
    feat = np.hstack([hist, [np.mean(f_gabor), np.std(f_gabor), sharp, noise, m_val[0][0], s_val[0][0]]]).reshape(1, -1)
    
    return feat, img, canny, {"LBP": np.var(hist)*1000, "Gabor": np.std(f_gabor), "Canny": np.sum(canny)/(128*128), "Sharp": sharp}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2fe; text-align:center;'>👁️ NEXUS-EYE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:10px; color:#555;'>V 3.2 PREMIUM FORENSIC</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("📁 **Unité de Contrôle**")
    st.info("Système calibré pour l'analyse fréquentielle des textures épidermiques.")
    st.markdown("---")
    st.write("🛠️ **Méthodes actives :**")
    st.caption("- Local Binary Patterns")
    st.caption("- Gabor Wavelets")
    st.caption("- Canny Edge Detection")
    st.caption("- Laplacian Variance")

# --- MAIN ---
st.markdown('<h1 class="main-header">NEXUS-EYE TERMINAL</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BEYOND VISUAL PERCEPTION</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    feat, original, canny_img, m = get_analysis(file_bytes)
    
    col_vis, col_data = st.columns([1.2, 1])

    with col_vis:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="SCAN SOURCE RGB", use_container_width=True)
        st.image(canny_img, caption="ANALYSE STRUCTURELLE (CANNY)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_data:
        st.markdown("<p style='font-weight:bold; color:#00f2fe;'>📊 TÉLÉMÉTRIE DE PRÉCISION</p>", unsafe_allow_html=True)
        
        # Grid de métriques
        row1 = st.columns(2)
        with row1[0]:
            st.markdown(f'<div class="metric-box"><span class="m-label">Porosité LBP</span><span class="m-val">{m["LBP"]:.4f}</span><p class="m-desc">Détecte le lissage artificiel des pores. L\'IA tend à uniformiser les textures.</p></div>', unsafe_allow_html=True)
        with row1[1]:
            st.markdown(f'<div class="metric-box"><span class="m-label">Orientation Gabor</span><span class="m-val">{m["Gabor"]:.4f}</span><p class="m-desc">Vérifie la cohérence des traits (rides, poils). Analyse les fréquences biologiques.</p></div>', unsafe_allow_html=True)
        
        st.write("")
        row2 = st.columns(2)
        with row2[0]:
            st.markdown(f'<div class="metric-box"><span class="m-label">Densité Bords</span><span class="m-val">{m["Canny"]:.4f}</span><p class="m-desc">Analyse la netteté des contours. L\'IA génère souvent des bords trop mathématiques.</p></div>', unsafe_allow_html=True)
        with row2[1]:
            st.markdown(f'<div class="metric-box"><span class="m-label">Score Netteté</span><span class="m-val">{m["Sharp"]:.1f}</span><p class="m-desc">Identifie le flou de synthèse. Les images réelles possèdent un bruit de capteur unique.</p></div>', unsafe_allow_html=True)

        if st.button("EXÉCUTER LE DIAGNOSTIC"):
            with st.spinner("CONFRONTATION..."):
                time.sleep(1.5)
                model = joblib.load('ia_human_detector.pkl')
                scaler = joblib.load('scaler_human.pkl')
                
                f_scaled = scaler.transform(feat)
                pred = model.predict(f_scaled)[0]
                prob = model.predict_proba(f_scaled)[0]

                st.markdown('<div class="verdict-box">', unsafe_allow_html=True)
                if pred == 1:
                    st.markdown(f"<h2 style='color:#ff4b4b; margin:0;'>🚨 SIGNATURE IA DÉTECTÉE</h2>", unsafe_allow_html=True)
                    st.write(f"Indice de fraude : {prob[1]*100:.2f}%")
                else:
                    st.markdown(f"<h2 style='color:#00ff88; margin:0;'>✅ SUJET RÉEL IDENTIFIÉ</h2>", unsafe_allow_html=True)
                    st.write(f"Confiance biologique : {prob[0]*100:.2f}%")

                st.markdown('</div>', unsafe_allow_html=True)
