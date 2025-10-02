# Contador de Pessoas em Tempo Real (Python, YOLOv8 + ByteTrack)

Projeto em Python para **detecção e contagem de pessoas em tempo real** a partir de **webcam, arquivos de vídeo ou streams**.  
Utiliza **YOLOv8** (Ultralytics) e **ByteTrack** (via [Supervision](https://github.com/roboflow/supervision)) para identificar pessoas, rastrear IDs e gerar métricas de fluxo e ocupação.  

---

## ✨ Funcionalidades
- Detecção de pessoas por frame com YOLOv8 (classe `person`).
- Rastreamento multi-objeto (ByteTrack) com IDs estáveis.
- Dois modos de contagem:
  - **Cruzamento de Linha:** Contagem de entradas e saídas (A→B / B→A) através de uma linha configurável.
  - **Área de Interesse (ROI):** Monitoramento da ocupação atual e IDs únicos dentro de uma região poligonal.
- Visualização em tempo real com OpenCV, exibindo bounding boxes, IDs, trilhas e zonas de contagem.
- Logs periódicos em JSON para `stdout` e exportação opcional em CSV.
- Resumo final das contagens em JSON ao encerrar a aplicação.
- Cálculo e exibição do FPS de processamento.

---

## 📂 Estrutura do Projeto
```
PeopleCounter/
├── .gitignore
├── Makefile            # atalhos: run/test/lint
├── README.md           # este arquivo
├── pyproject.toml      # configs black, isort, ruff
├── requirements.txt    # dependências Python
├── src/
│   ├── app.py          # Ponto de entrada principal da aplicação
│   ├── detect.py       # Lógica de detecção de pessoas com YOLOv8
│   ├── track.py        # Lógica de rastreamento de objetos com ByteTrack
│   ├── zones.py        # Definição e lógica das zonas de contagem (linha e ROI)
│   ├── draw.py         # Funções para anotação visual no frame
│   └── metrics.py      # Utilitários para telemetria (JSON/CSV)
├── tests/              # Testes automatizados (unitários e de integração)
│   ├── test_zones.py
│   ├── test_metrics.py
│   └── test_pipeline_smoke.py
└── data/
    └── samples/        # Vídeos de teste (com .gitkeep)
```

---

## 🚀 Instalação

### Pré-requisitos
- Python **3.10+** (recomendado: 3.10.11).
- `pip` atualizado (`python -m pip install --upgrade pip`).
- **Em Linux (Ubuntu/Debian):** Pode ser necessário instalar `ffmpeg` e `libgl1`:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libgl1
  ```
- **Em Windows:** Recomenda-se o uso do **WSL2** com Ubuntu para melhor compatibilidade.

### Passo a passo
1. Clone o repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/people-counter.git
   cd people-counter
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # .venv\Scripts\activate    # Windows (no prompt de comando)
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   pip install ruff black  # Ferramentas de linting e formatação
   ```

---

## ▶️ Uso

Para executar a aplicação, utilize o comando `python -m src.app` a partir da raiz do projeto. Pressione **`q`** com a janela de vídeo em foco para encerrar a aplicação.

### Argumentos da Linha de Comando

- `--source <caminho>`: Fonte de vídeo. Pode ser `0` (webcam), o caminho para um arquivo de vídeo (ex: `data/samples/video.mp4`) ou uma URL de stream. Padrão: `0`.
- `--model <nome_modelo>`: Nome do arquivo de pesos do modelo YOLO (ex: `yolov8n.pt`, `yolov8s.pt`). Padrão: `yolov8n.pt`.
- `--conf <float>`: Limiar de confiança para detecção (0.0 a 1.0). Detecções abaixo deste valor são ignoradas. Padrão: `0.35`.
- `--iou <float>`: Limiar de Intersection over Union (IoU) para Non-Maximum Suppression (NMS). Padrão: `0.45`.
- `--device <string>`: Dispositivo para inferência (ex: `cpu`, `cuda:0`). Padrão: `None` (auto-seleção).
- `--imgsz <int>`: Tamanho da imagem para inferência em pixels (ex: `640`). Padrão: `640`.
- `--csv <caminho_arquivo>`: Opcional. Caminho para um arquivo CSV onde as métricas serão logadas a cada segundo.
- `--out <caminho_arquivo>`: Opcional. Caminho para um arquivo JSON onde um resumo final das contagens será salvo ao encerrar a aplicação.
- `--line <x1 y1 x2 y2>`: Opcional. Coordenadas da linha para contagem de cruzamentos (4 valores: x1 y1 x2 y2). Se não especificado, usa linha horizontal no meio da tela.
- `--roi <x1 y1 x2 y2 ...>`: Opcional. Coordenadas do polígono ROI (mínimo 6 valores para 3 pontos). Se não especificado, usa retângulo centralizado.
- `--headless`: Opcional. Executa sem interface gráfica (sem cv2.imshow). Requer `--output-video`.
- `--output-video <caminho_arquivo>`: Opcional. Caminho para salvar o vídeo anotado de saída.

### Exemplos de Execução

1.  **Usando a webcam (padrão):**
    ```bash
    python -m src.app
    ```

2.  **Usando um arquivo de vídeo:**
    ```bash
    python -m src.app --source data/samples/my_video.mp4
    ```
    *Dica: Coloque seus próprios vídeos na pasta `data/samples/` para fácil acesso.* 

3.  **Com contagem por linha (exemplo de linha horizontal no meio da tela):**
    ```bash
    python -m src.app --source 0 --line 0 360 1280 360
    ```
    *(Ajuste os pontos `x1 y1 x2 y2` conforme a resolução da sua câmera/vídeo.)*

4.  **Com Área de Interesse (ROI) (exemplo de ROI retangular):**
    ```bash
    python -m src.app --source 0 --roi 100 100 540 100 540 380 100 380
    ```
    *(Ajuste os pontos `x1 y1 ...` conforme a resolução da sua câmera/vídeo.)*

5.  **Logando métricas para CSV e salvando resumo final:**
    ```bash
    python -m src.app --source 0 --csv metrics.csv --out final_summary.json
    ```

6.  **Usando o modelo `yolov8s.pt` com maior confiança:**
    ```bash
    python -m src.app --source 0 --model yolov8s.pt --conf 0.50
    ```

7.  **Modo headless (sem GUI) com exportação de vídeo:**
    ```bash
    python -m src.app --source data/samples/video.mp4 --headless --output-video output.mp4
    ```
    *Útil para processamento em servidores sem interface gráfica.*

---

## ⚠️ Troubleshooting

-   **Webcam não abre:** Verifique se a câmera está conectada e se você concedeu as permissões necessárias ao aplicativo (no macOS, por exemplo, Python precisa de permissão de câmera). Tente usar `--source 1`, `--source 2`, etc., caso tenha múltiplas câmeras.
-   **Baixo FPS:**
    -   Reduza a resolução da câmera/vídeo.
    -   Diminua o valor de `--conf` (pode aumentar falsos positivos).
    -   Aumente o valor de `--iou` (pode agrupar detecções).
    -   Considere usar um modelo menor (ex: `yolov8n.pt` é mais rápido que `yolov8s.pt`).
    -   Se disponível, utilize uma GPU (`--device cuda:0`).
-   **Janela preta/aplicativo não inicia em servidor headless:** A aplicação requer uma interface gráfica para exibir o vídeo. Para ambientes sem GUI, você precisaria modificar o código para desabilitar `cv2.imshow` ou exportar o vídeo anotado para um arquivo (funcionalidade futura).

---

## 🛠️ Tecnologias
- [Python 3.10+](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack e utilitários)](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [Pytest](https://docs.pytest.org/)
- [Ruff](https://docs.astral.sh/ruff/) (linter)
- [Black](https://github.com/psf/black) (formatador de código)

---

## 📄 Licença
Este projeto é distribuído sob a licença **MIT**.  

**Importante:** O pacote `ultralytics` (que inclui os modelos YOLOv8) é licenciado sob **AGPL-3**. Isso significa que qualquer software que utilize ou seja derivado do `ultralytics` e seja distribuído publicamente, deve também ser licenciado sob AGPL-3. Avalie as implicações desta licença para o seu caso de uso específico.

---


# Real-time People Counter (Python, YOLOv8 + ByteTrack)

Python project for **real-time people detection and counting** from **webcam, video files, or streams**.  
It uses **YOLOv8** (Ultralytics) and **ByteTrack** (via [Supervision](https://github.com/roboflow/supervision)) to identify people, track IDs, and generate flow and occupancy metrics.  

---

## ✨ Features
- Person detection per frame with YOLOv8 (`person` class).
- Multi-object tracking (ByteTrack) with stable IDs.
- Two counting modes:
  - **Line Crossing:** Counts entries and exits (A→B / B→A) across a configurable line.
  - **Region of Interest (ROI):** Monitors current occupancy and unique IDs within a polygonal region.
- Real-time visualization with OpenCV, displaying bounding boxes, IDs, traces, and counting zones.
- Periodic JSON logs to `stdout` and optional CSV export.
- Final summary of counts in JSON upon application termination.
- Processing FPS calculation and display.

---

## 📂 Project Structure
```
PeopleCounter/
├── .gitignore
├── Makefile            # shortcuts: run/test/lint
├── README.md           # this file
├── pyproject.toml      # black, isort, ruff configs
├── requirements.txt    # Python dependencies
├── src/
│   ├── app.py          # Main application entry point
│   ├── detect.py       # Person detection logic with YOLOv8
│   ├── track.py        # Object tracking logic with ByteTrack
│   ├── zones.py        # Definition and logic of counting zones (line and ROI)
│   ├── draw.py         # Functions for visual annotation on the frame
│   └── metrics.py      # Utilities for telemetry (JSON/CSV)
├── tests/              # Automated tests (unit and integration)
│   ├── test_zones.py
│   ├── test_metrics.py
│   └── test_pipeline_smoke.py
└── data/
    └── samples/        # Test videos (with .gitkeep)
```

---

## 🚀 Installation

### Prerequisites
- Python **3.10+** (recommended: 3.10.11).
- Updated `pip` (`python -m pip install --upgrade pip`).
- **On Linux (Ubuntu/Debian):** `ffmpeg` and `libgl1` might be required:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libgl1
  ```
- **On Windows:** **WSL2** with Ubuntu is recommended for better compatibility.

### Step-by-step
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/people-counter.git
   cd people-counter
   ```

2. Create and activate the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # .venv\Scripts\activate    # Windows (in command prompt)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install ruff black  # Linting and formatting tools
   ```

---

## ▶️ Usage

To run the application, use the command `python -m src.app` from the project root. Press **`q`** with the video window in focus to terminate the application.

### Command Line Arguments

- `--source <path>`: Video source. Can be `0` (webcam), the path to a video file (e.g., `data/samples/video.mp4`), or a stream URL. Default: `0`.
- `--model <model_name>`: Name of the YOLO model weights file (e.g., `yolov8n.pt`, `yolov8s.pt`). Default: `yolov8n.pt`.
- `--conf <float>`: Confidence threshold for detection (0.0 to 1.0). Detections below this value are ignored. Default: `0.35`.
- `--iou <float>`: Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). Default: `0.45`.
- `--device <string>`: Device for inference (e.g., `cpu`, `cuda:0`). Default: `None` (auto-selection).
- `--imgsz <int>`: Image size for inference in pixels (e.g., `640`). Default: `640`.
- `--csv <file_path>`: Optional. Path to a CSV file where metrics will be logged every second.
- `--out <file_path>`: Optional. Path to a JSON file where a final summary of counts will be saved upon application termination.
- `--line <x1 y1 x2 y2>`: Optional. Line coordinates for crossing counter (4 values: x1 y1 x2 y2). If not specified, uses horizontal line at screen center.
- `--roi <x1 y1 x2 y2 ...>`: Optional. ROI polygon coordinates (minimum 6 values for 3 points). If not specified, uses centered rectangle.
- `--headless`: Optional. Run without GUI (no cv2.imshow). Requires `--output-video`.
- `--output-video <file_path>`: Optional. Path to save annotated output video.

### Execution Examples

1.  **Using the webcam (default):**
    ```bash
    python -m src.app
    ```

2.  **Using a video file:**
    ```bash
    python -m src.app --source data/samples/my_video.mp4
    ```
    *Tip: Place your own videos in the `data/samples/` folder for easy access.* 

3.  **With line counting (example of a horizontal line in the middle of the screen):**
    ```bash
    python -m src.app --source 0 --line 0 360 1280 360
    ```
    *(Adjust `x1 y1 x2 y2` points according to your camera/video resolution.)*

4.  **With Region of Interest (ROI) (example of a rectangular ROI):**
    ```bash
    python -m src.app --source 0 --roi 100 100 540 100 540 380 100 380
    ```
    *(Adjust `x1 y1 ...` points according to your camera/video resolution.)*

5.  **Logging metrics to CSV and saving final summary:**
    ```bash
    python -m src.app --source 0 --csv metrics.csv --out final_summary.json
    ```

6.  **Using the `yolov8s.pt` model with higher confidence:**
    ```bash
    python -m src.app --source 0 --model yolov8s.pt --conf 0.50
    ```

7.  **Headless mode (no GUI) with video export:**
    ```bash
    python -m src.app --source data/samples/video.mp4 --headless --output-video output.mp4
    ```
    *Useful for processing on servers without a graphical interface.*

---

## ⚠️ Troubleshooting

-   **Webcam not opening:** Check if the camera is connected and if you have granted the necessary permissions to the application (on macOS, for example, Python needs camera permission). Try using `--source 1`, `--source 2`, etc., if you have multiple cameras.
-   **Low FPS:**
    -   Reduce camera/video resolution.
    -   Decrease the `--conf` value (may increase false positives).
    -   Increase the `--iou` value (may group detections).
    -   Consider using a smaller model (e.g., `yolov8n.pt` is faster than `yolov8s.pt`).
    -   If available, use a GPU (`--device cuda:0`).
-   **Black window/application not starting on headless server:** The application requires a graphical interface to display the video. For environments without a GUI, you would need to modify the code to disable `cv2.imshow` or export the annotated video to a file (future functionality).

---

## 🛠️ Technologies
- [Python 3.10+](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack and utilities)](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [Pytest](https://docs.pytest.org/)
- [Ruff](https://docs.astral.sh/ruff/) (linter)
- [Black](https://github.com/psf/black) (code formatter)

---

## 📄 License
This project is distributed under the **MIT** license.  

**Important:** The `ultralytics` package (which includes YOLOv8 models) is licensed under **AGPL-3**. This means that any software that uses or is derived from `ultralytics` and is publicly distributed must also be licensed under AGPL-3. Evaluate the implications of this license for your specific use case.

---
