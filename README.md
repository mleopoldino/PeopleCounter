# Contador de Pessoas em Tempo Real (Python, YOLOv8 + ByteTrack)

Projeto em Python para **detecção e contagem de pessoas em tempo real** a partir de **webcam, arquivos de vídeo ou streams**.  
Utiliza **YOLOv8** (Ultralytics) e **ByteTrack** (via [Supervision](https://github.com/roboflow/supervision)) para identificar pessoas, rastrear IDs e gerar métricas de fluxo e ocupação.  

---

## ✨ Funcionalidades
- Detecção de pessoas por frame com YOLOv8.
- Rastreamento multi-objeto (ByteTrack).
- Dois modos de contagem:
  - **Cruzamento de Linha:** entradas e saídas (A→B / B→A).
  - **Área de Interesse (ROI):** ocupação atual e IDs únicos.
- Visualização em tempo real com OpenCV.
- Logs periódicos em JSON e exportação opcional em CSV.
- Estrutura preparada para evolução futura (exportar vídeo, API REST, dashboards).

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
│   └── app.py          # ponto de entrada inicial
├── tests/
│   └── test_dummy.py   # teste placeholder (pytest)
└── data/
    └── samples/        # vídeos de teste (com .gitkeep)
```

---

## 🚀 Instalação

### Pré-requisitos
- Python **3.10** ou **3.11** (recomendado: 3.10.11).
- pip atualizado (`python -m pip install --upgrade pip`).
- Em Linux: pode ser necessário instalar `ffmpeg` e `libgl1`:
  ```bash
  sudo apt-get install -y ffmpeg libgl1
  ```

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
   # .venv\Scripts\activate    # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Uso

Rodar o aplicativo de exemplo (abre webcam e fecha com **q**):
```bash
make run
```

Rodar diretamente:
```bash
python src/app.py --source 0
```

Rodar os testes:
```bash
make test
```

Rodar lint (se instalar `black` e `ruff`):
```bash
make lint
```

---

## 🛠️ Tecnologias
- [Python 3.10+](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack e utilitários)](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [Pytest](https://docs.pytest.org/)

---

## 📌 Roadmap
- [ ] Implementar detecção de pessoas (YOLOv8).
- [ ] Adicionar tracking (ByteTrack).
- [ ] Implementar contagem por linha e ROI.
- [ ] Exportar métricas em CSV e JSON.
- [ ] Criar testes unitários e de integração.
- [ ] Documentar resultados e performance.
- [ ] Evoluir para exportação de vídeo anotado.
- [ ] (Opcional) API REST para métricas em tempo real.

---

## 📄 Licença
Este projeto é distribuído sob a licença **MIT**.  
Os modelos YOLOv8 seguem a licença **AGPL-3** da Ultralytics — revisar antes de uso comercial.

---

