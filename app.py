
import streamlit as st
import cv2
from deepface import DeepFace
import os
import glob
import time

# Configuração da página
st.set_page_config(page_title="Auditoria Facial - PIBITI", page_icon="🖥️", layout="wide")

st.title("🖥️ Sistema de Reconhecimento Facial e Auditoria (Zero-Trust)")
st.markdown("---")

# =====================================================================
# CONFIGURAÇÃO DE AMBIENTE (NUVEM)
# =====================================================================
pasta_testes = "testes"
pasta_banco_dados = "banco_faces"

# Garante que as pastas existam no servidor nuvem
os.makedirs(pasta_testes, exist_ok=True)
os.makedirs(pasta_banco_dados, exist_ok=True)

# Barra Lateral
st.sidebar.header("⚙️ Parâmetros do Sistema")
LIMITE_DE_SEGURANCA = st.sidebar.slider("Limiar de Segurança (Similaridade Mínima)", 0.10, 0.90, 0.50, 0.05)
margem = st.sidebar.slider("Margem Visual do Quadrado (Pixels)", 0, 100, 20, 5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dica:** O limiar padrão de 0.50 equilibra a Taxa de Falso Aceite (FAR) e Falsa Rejeição (FRR).")

# Interface de Status
st.subheader("📊 Status do Banco de Dados de Referência")
total_banco = len(glob.glob(os.path.join(pasta_banco_dados, "*.jpg"))) + len(glob.glob(os.path.join(pasta_banco_dados, "*.png")))
st.write(f"• **Identidades cadastradas no servidor:** {total_banco}")

# =====================================================================
# FUNÇÃO CENTRAL DE RECONHECIMENTO (ARCFACE + RETINAFACE)
# =====================================================================
def processar_frame_deepface(img_frame):
    rostos_processados = 0
    try:
        resultados = DeepFace.find(
            img_path=img_frame, 
            db_path=pasta_banco_dados, 
            model_name="ArcFace", 
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=True, 
            threshold=2.0,
            silent=True
        )
        
        for df_rosto in resultados:
            if df_rosto.empty: continue
            melhor_match = df_rosto.iloc[0]
            
            x, y, w, h = int(melhor_match['source_x']), int(melhor_match['source_y']), int(melhor_match['source_w']), int(melhor_match['source_h'])
            if w < 30 or h < 30: continue
            rostos_processados += 1
            
            similaridade = 1 - melhor_match['distance']
            
            if similaridade >= LIMITE_DE_SEGURANCA:
                identidade = os.path.basename(melhor_match['identity']).split('.')[0].upper()
                cor = (0, 255, 0)
                texto = f"AUTORIZADO: {identidade} (Simil: {similaridade:.2f})"
            else:
                cor = (0, 0, 255)
                texto = f"INTRUSO (Simil: {similaridade:.2f})"
                
            x1, y1 = max(0, x - margem), max(0, y - margem)
            x2, y2 = min(img_frame.shape[1], x + w + margem), min(img_frame.shape[0], y + h + margem)
            
            tamanho_fonte = max(0.5, w / 200.0)
            espessura = max(1, int(w / 100))

            cv2.rectangle(img_frame, (x1, y1), (x2, y2), cor, espessura + 1)
            posicao_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
            cv2.putText(img_frame, texto, (x1, posicao_y), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, (0, 0, 0), espessura + 2, cv2.LINE_AA)
            cv2.putText(img_frame, texto, (x1, posicao_y), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, cor, espessura, cv2.LINE_AA)
            
    except Exception as e:
        st.exception(e)
        
    return img_frame, rostos_processados

# =====================================================================
# BLOCO 1: PROCESSAMENTO EM LOTE (VIA UPLOAD)
# =====================================================================
st.markdown("---")
st.subheader("📁 Auditoria em Lote de Imagens")
arquivos_lote = st.file_uploader("Envie imagens para auditar simultaneamente:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if arquivos_lote:
    if st.button("🚀 Processar Lote"):
        tempo_total_inicio = time.time()
        rostos_totais_processados = 0
        
        # Limpa o cache do DeepFace para não usar parâmetros antigos do slider
        for pkl in glob.glob(os.path.join(pasta_banco_dados, "*.pkl")):
            os.remove(pkl)
            
        progresso = st.progress(0)
        
        for idx, arquivo in enumerate(arquivos_lote):
            # Salva o arquivo temporário enviado pelo usuário
            caminho_temp = os.path.join(pasta_testes, arquivo.name)
            with open(caminho_temp, "wb") as f:
                f.write(arquivo.read())
                
            progresso.progress((idx + 1) / len(arquivos_lote))
            
            img = cv2.imread(caminho_temp)
            img_processada, rostos_frame = processar_frame_deepface(img)
            rostos_totais_processados += rostos_frame
            
            img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Resultado: {arquivo.name}", use_column_width=True)
            
            # Limpa o arquivo temp
            os.remove(caminho_temp)

        duracao_total = time.time() - tempo_total_inicio
        st.success(f"🏁 Lote concluído em {duracao_total:.2f}s! ({rostos_totais_processados} rostos auditados).")

# =====================================================================
# BLOCO 2: VÍDEO GRAVADO (VIA UPLOAD)
# =====================================================================
st.markdown("---")
st.subheader("📼 Auditoria em Vídeo Gravado")
arquivo_video = st.file_uploader("Selecione um vídeo (.mp4, .avi):", type=["mp4", "avi"])

if arquivo_video is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(arquivo_video.read())
        
    if st.button("▶️ Iniciar Análise do Vídeo"):
        video_placeholder = st.empty()
        cap = cv2.VideoCapture("temp_video.mp4")
        
        pular_quadros = 5 
        contador = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("✅ Fim do vídeo alcançado!")
                break
                
            contador += 1
            if contador % pular_quadros != 0: continue
            
            frame_processado, _ = processar_frame_deepface(frame)
            frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()
        if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")