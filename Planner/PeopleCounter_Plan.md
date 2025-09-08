
# Plano de Implementação — Contador de Pessoas em Tempo Real (Python, OSS)

**Versão:** 1.0  
**Autor:** Marcelo (orientação) — execução por *Gemini CLI* e *Codex* com base neste plano  
**Objetivo do documento:** detalhar requisitos, arquitetura, tarefas, prompts e critérios de aceite para que *Gemini CLI* e *Codex* codifiquem **100%** da solução seguindo exatamente as instruções abaixo.

---

## 0) Feedback sobre a ideia (avaliando o desejo)
A solução de **contador de pessoas em tempo real** é **altamente válida e pragmática** para provar valor de IA de visão computacional com uso 100% open‑source. Ela se conecta facilmente a cenários reais (filas, ocupação, segurança), possui **baixo atrito técnico** (roda em CPU com fps moderado; em GPU escala bem) e tem **superfície de risco controlada** (não faz identificação/biometria). É um ótimo candidato para POCs rápidas, com resultados mensuráveis e material repetível para demos clientes NTConsult.

---

## 1) Visão & Escopo

### 1.1 Visão
Construir um **aplicativo Python** que conte **pessoas** em tempo real a partir de **webcam, arquivo de vídeo ou stream** e entregue **métricas operacionais** (ex.: entradas/saídas numa linha, ocupação em uma área), com **visualização** e **logs** para análise.

### 1.2 Escopo do MVP
- **Entrada:** webcam (`--source 0`), vídeo local (`--source video.mp4`) ou URL compatível com OpenCV.
- **Saída:** janela com vídeo anotado; impressão **JSON**/s no stdout a cada segundo; **CSV opcional**.
- **Modos de contagem:** (a) **Cruzamento de linha** A→B/B→A; (b) **Ocupação de ROI (polígono)** (presentes e únicos).
- **Modelo:** detector **YOLOv8n** (Ultralytics) + *tracker* **ByteTrack** via `supervision`.
- **Sem Camunda** nesta fase; API REST **opcional** somente se explicitado em backlog.

### 1.3 Fora de escopo (por agora)
- Reconhecimento de identidade/face.  
- Persistência em banco e dashboards.  
- Multi-câmera simultânea (pode entrar como evolução).

---

## 2) Requisitos

### 2.1 Funcionais (RF)
1. RF-01: Carregar fonte de vídeo (webcam/arquivo/URL).
2. RF-02: Detectar pessoas por frame usando YOLOv8 (classe COCO `person`).
3. RF-03: Fazer *tracking* multi‑objeto (ByteTrack) e manter IDs estáveis.
4. RF-04: **Contagem por linha**: calcular `in` (A→B) e `out` (B→A).
5. RF-05: **Contagem por ROI**: `roi_current` (presentes) e `roi_unique` (IDs distintos observados).
6. RF-06: Desenhar bounding boxes, rótulos (id/conf), trilhas, linha e/ou polígono.
7. RF-07: Emitir **logs JSON** 1x/s com `line_in`, `line_out`, `roi_current`, `roi_unique`, `fps_cap`.
8. RF-08: Se habilitado `--csv`, **acrescentar** métricas por segundo em um arquivo CSV (com cabeçalho).
9. RF-09: Encerrar limpo com `q` e salvar resumo final se `--out counts.json` for usado.
10. RF-10: Parâmetros por CLI: `--source`, `--line x1 y1 x2 y2`, `--roi x1 y1 ...`, `--conf`, `--iou`, `--device`, `--max_fps`, `--csv`, `--out`.

### 2.2 Não Funcionais (RNF)
- RNF-01: Rodar em **CPU** (>= 10–15 fps em 720p com `yolov8n` e `--max_fps 20–30` em máquinas modernas).
- RNF-02: Suportar **GPU** opcional (`--device cuda:0`).
- RNF-03: Código **modular**, legível, com *type hints* e docstrings.
- RNF-04: Testes automatizados (unidade e integração leve) executáveis via `pytest`.
- RNF-05: Logs sem dados sensíveis; não armazenar vídeo por padrão.
- RNF-06: Dependências com **version pinning** para reprodutibilidade.
- RNF-07: Licenças OSS: atenção ao **AGPL-3** do pacote `ultralytics` (avaliar compatibilidade para uso interno/demo).

### 2.3 Restrições & Premissas
- Ambiente alvo: macOS, Linux, Windows (preferencialmente WSL no Windows).
- Câmeras reais podem ter latência/variação; o MVP deve tolerar queda de FPS.
- Privacidade: sem biometria; armazenar somente métricas.

---

## 3) Arquitetura de Referência

### 3.1 Componentes
- **InputAdapter:** OpenCV VideoCapture (webcam/arquivo/URL).
- **Detector:** YOLOv8n (Ultralytics) filtrando classe `person`.
- **Tracker:** ByteTrack (`supervision.ByteTrack`).
- **Counters:** 
  - `LineZone` (A→B/B→A);
  - `PolygonZone` (presentes/únicos).
- **Annotators:** caixas, labels, trilhas, linha/ROI.
- **Telemetry:** emissor JSON por segundo; *CSV appender*.
- **CLI:** argparse para todos os parâmetros.

### 3.2 Fluxo (alto nível)
```
Frame -> Detector -> Detections(person) -> Tracker ->
  -> LineZone.trigger() -> in/out
  -> PolygonZone.trigger() -> roi_current/unique
  -> Annotate -> Display
  -> Every 1s: print JSON (+ CSV se habilitado)
```

### 3.3 Contratos de Saída (logs)
**JSON por segundo (stdout):**
```json
{
  "line_in": 12,
  "line_out": 7,
  "roi_current": 3,
  "roi_unique": 19,
  "fps_cap": 30
}
```
**CSV (se `--csv metrics.csv`):**
```
ts,line_in,line_out,roi_current,roi_unique,fps_cap
1725811001,12,7,3,19,30
...
```

---

## 4) Stack Tecnológico & Licenças
- **Python 3.10+**
- **ultralytics (YOLOv8)** — *AGPL-3*. Uso interno/POC ok; avaliar implicações se houver distribuição comercial de código derivado.
- **supervision** — utilitários (ByteTrack, zonas, anotações).
- **opencv-python**, **numpy**
- (Opcional) **torch**/**torchvision** com CUDA.
> _Nota sobre licenças_: para produção comercial com código fechado, considerar alternativas de detecção com licenças permissivas ou contratar licença comercial apropriada.

---

## 5) Setup do Ambiente

### 5.1 Pré‑requisitos de SO
- **macOS:** Homebrew instalado (para ffmpeg opcional).  
- **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg libgl1`  
- **Windows:** preferir **WSL2** com Ubuntu + acesso à webcam (ou usar fonte de vídeo por arquivo).

### 5.2 Passos
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics==8.3.37 supervision==0.23.0 opencv-python>=4.9 numpy>=1.24
# GPU opcional: instalar torch compatível com sua CUDA
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.3 Verificação rápida
```python
python - << 'PY'
import cv2, ultralytics, supervision
print("cv2:", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
print("supervision ok")
PY
```

---

## 6) Estrutura do Projeto
```
people-counter/
  src/
    app.py                 # ponto de entrada CLI
    video_io.py            # adaptadores de entrada
    detect.py              # wrapper YOLO
    track.py               # wrapper ByteTrack
    zones.py               # linha e ROI helpers
    draw.py                # anotações (caixas, labels, trilhas)
    metrics.py             # JSON/CSV logging por segundo
    config.py              # parser de CLI e validações
  tests/
    test_zones.py
    test_metrics.py
    test_pipeline_smoke.py
  data/
    samples/               # vídeos curtos de teste (colocar depois)
  requirements.txt
  README.md
  Makefile                 # targets comuns (run/test/lint)
  pyproject.toml           # black/isort/ruff config (estilo opcional)
```

---

## 7) Plano Passo a Passo (WBS com Prompts)

> **Como usar os prompts:** copie e cole no **Gemini CLI** e no **Codex**. Sempre inclua o **contexto do repositório**, o **RF/RNF** e os **critérios de aceite** da tarefa. Peça **arquivos completos** e **testes** quando aplicável.

### Sprint 0 — Bootstrap & Qualidade
**Objetivo:** scaffold do projeto, lint/test infra e `app.py` mínimo (abre fonte, fecha com `q`).  
**Tarefas:**
1. Criar estrutura de pastas (ver §6) + `requirements.txt` com pins.
2. Implementar `config.py` (argparse com: `--source`, `--line`, `--roi`, `--conf`, `--iou`, `--device`, `--max_fps`, `--csv`, `--out`).
3. `app.py` abre `VideoCapture`, mostra janela e fecha com `q`.
4. Adicionar `pyproject.toml` com `ruff`/`black` (opcional) e `pytest`.
5. `Makefile` com `run`, `test`, `lint`.
**Critérios de aceite (DoD):**
- `make run` abre webcam (`--source 0`) e fecha com `q`.
- `pytest` executa e passa (mesmo com testes placeholder).

**Prompt sugerido (Gemini/Codex):**
> *“Atue como engenheiro sênior. Crie o scaffold descrito (pastas/arquivos), com `app.py` que abre um vídeo a partir de `--source` e fecha com `q`. Implemente `config.py` com validação básica. Gere `requirements.txt` pinado. Inclua `pytest` com 2 testes simples (ex.: parse de args e função dummy). Use type hints e docstrings.”*

---

### Sprint 1 — Detecção (YOLO) e Filtro `person`
**Objetivo:** detectar pessoas por frame.  
**Tarefas:**
1. `detect.py`: classe `PersonDetector` com `predict(frame) -> Detections` (bbox, conf, class_id).
2. Baixar peso `yolov8n.pt` na primeira execução (a lib já gerencia).
3. Filtrar apenas classe `person` (id 0 COCO).
**DoD:**
- `app.py` imprime nº de pessoas detectadas por frame no console (sem tracking ainda).
- `--conf` e `--iou` ajustáveis.
**Prompt:**
> *“Implemente `detect.py` com classe `PersonDetector` usando Ultralytics YOLOv8n. Método `predict(frame)` retorna somente detecções da classe `person` com (xyxy, conf). Adapte `app.py` para mostrar em tempo real a contagem de pessoas detectadas por frame.”*

---

### Sprint 2 — Tracking (ByteTrack) e IDs estáveis
**Objetivo:** manter IDs por pessoa ao longo do tempo.  
**Tarefas:**
1. `track.py`: wrapper para `supervision.ByteTrack` com `update(detections)->detections_tracked` (adiciona `tracker_id`).
2. Integrar no `app.py`: fluxo `detect -> track`.
**DoD:**
- Exibir no console `len(detections_tracked)` e alguns `tracker_id`.
**Prompt:**
> *“Implemente `track.py` integrando `supervision.ByteTrack`. Exponha `update(detections)` retornando detecções com `tracker_id`. Ajuste `app.py` para imprimir IDs por frame.”*

---

### Sprint 3 — Zonas de Contagem (Linha & ROI)
**Objetivo:** contar `line_in/out` e `roi_current/unique`.  
**Tarefas:**
1. `zones.py`: helpers para criar `LineZone` e `PolygonZone` (`supervision`) a partir dos args.
2. Em `app.py`, acionar `line_zone.trigger(detections)` e `polygon_zone.trigger(detections)`.
3. Manter conjunto `seen_ids_in_roi`.
**DoD:**
- A cada segundo (timer), imprimir **JSON** com métricas.
**Prompt:**
> *“Implemente `zones.py` com funções `make_line_zone(p1,p2)` e `make_polygon_zone(points, wh)`. Integre no `app.py` para calcular contagens e imprimir JSON a cada 1s com `line_in`, `line_out`, `roi_current`, `roi_unique`.”*

---

### Sprint 4 — Visualização (Anotações)
**Objetivo:** desenhar caixas, labels (id/conf), trilhas, linha e ROI.  
**Tarefas:**
1. `draw.py`: usar `supervision` (`BoxAnnotator`, `LabelAnnotator`, `TraceAnnotator`, `LineZoneAnnotator`, `PolygonZoneAnnotator`).
2. Integrar no `app.py` ao final do pipeline.
**DoD:**
- Janela exibe corretamente boxes/labels/trilhas e indicações de linha/ROI.
**Prompt:**
> *“Implemente `draw.py` com funções para anotar frame (caixas, labels com id/conf, trilhas) e desenhar linha/ROI. Integre no `app.py` antes do `imshow`.”*

---

### Sprint 5 — Telemetria (JSON/CSV) e Resumo Final
**Objetivo:** métricas 1x/s + CSV opcional + `--out` JSON final.  
**Tarefas:**
1. `metrics.py`: utilitário com `emit_every_second(callback)` e `append_csv(path, row)`.
2. Integrar timer 1s no `app.py` para imprimir JSON; se `--csv`, gravar linha.
3. Ao sair, se `--out`, escrever resumo final (`line_in/out`, `roi_unique`, `frames`).
**DoD:**
- CSV criado com cabeçalho; JSON por segundo no stdout.
**Prompt:**
> *“Implemente `metrics.py` com funções para emissão por segundo e append em CSV. Integre no `app.py` para logs JSON 1x/s e CSV quando `--csv` informado, além de `--out` com resumo.”*

---

### Sprint 6 — Testes Automatizados
**Objetivo:** garantir corretude da geometria e do ciclo principal.  
**Tarefas:**
1. `test_zones.py`: testar **interseção** de trajetos com linha e **point-in-polygon** básico (mockar detecções).
2. `test_metrics.py`: testar formatação de JSON e escrita de CSV (com tmpdir).
3. `test_pipeline_smoke.py`: carregar `app` em modo *dry-run* com 10 frames sintéticos (ex.: imagens em preto) para validar ciclo sem erros.
**DoD:**
- `pytest -q` verde no CI local.
**Prompt:**
> *“Implemente testes conforme descrito: geometria (linha/ROI), CSV/JSON e smoke do pipeline. Use `pytest` e `tmp_path`.”*

---

### Sprint 7 — Performance & Tuning
**Objetivo:** garantir FPS alvo e estabilidade.  
**Tarefas:**
1. Medir FPS real com diferentes `--max_fps` (10/20/30) em 720p.
2. Ajustar `conf/iou` e ativar `model.fuse()` para ganho.
3. Documentar *trade-offs* `yolov8n` x `yolov8s`.
**DoD:**
- Tabela de resultados + recomendações de default.
**Prompt:**
> *“Crie um script `bench.md` e adicione medições de FPS sob diferentes parâmetros, recomendando valores padrão.”*

---

### Sprint 8 — Empacotamento & Documentação
**Objetivo:** docs claras e execução simples.  
**Tarefas:**
1. `README.md` completo (instalação, uso, exemplos com linha e ROI, troubleshooting).
2. (Opcional) `Dockerfile` com CUDA *base* e tag CPU.
3. Exemplos no `data/samples/` (colocar instruções para o usuário adicionar seus vídeos).
**DoD:**
- README reprodutível do zero.
**Prompt:**
> *“Escreva um README completo com instruções multiplataforma, argumentos CLI e exemplos. Inclua seção de problemas comuns.”*

---

## 8) Critérios de Aceite (gerais do projeto)
- Detector + tracker funcionais, com IDs estáveis na maioria dos cenários simples.
- Contagens corretas em cenários controlados (pelo menos 3 vídeos de teste curtos):
  - Linha horizontal com fluxo unidirecional.
  - ROI retangular simples com entradas e saídas claras.
  - Caso de oclusão moderada (duas pessoas cruzando).
- JSON por segundo correto; CSV criado e com cabeçalho único.
- Fechamento limpo sem “segfaults”/janelas travadas.
- Documentação suficiente para um novo dev reproduzir em <30 min.

---

## 9) Plano de Testes (detalhado)

### 9.1 Unit
- **Line crossing:** simular trajetórias de bounding boxes atravessando a linha (A→B e B→A) e validar contadores.
- **ROI:** simular entradas/saídas em polígono; validar `roi_current` e `roi_unique`.
- **CSV:** garantir cabeçalho e append correto.

### 9.2 Integração leve
- Pipeline com frames sintéticos (ruído/quadros pretos) para testar ciclo sem detecção, evitando exceções.
- Execução com vídeo curto real (10–20s) e verificação manual de contagens.

### 9.3 Performance
- Medir FPS médio, latência por frame e uso de CPU/RAM em diferentes parâmetros.
- Verificar queda de FPS com ROI complexas e `trace_length` alto.

### 9.4 Aceitação Manual
- Operador define uma **linha** clara; caminhar na frente da câmera 3 vezes A→B e 2 vezes B→A; comparar contagem.
- Definir **ROI** simples; entrar e sair 5 vezes; validar `unique >= 1` e variação de `current`.

---

## 10) Segurança, Privacidade & Ética
- Não armazenar vídeo por padrão; somente métricas numéricas.
- Não realizar identificação pessoal.
- Se for gravar para testes, manter vídeos localmente e com consentimento.
- Comunicar claramente limitações (oclusões, ângulos ruins, iluminação).

---

## 11) Operação & Runbook
- **Execução comum:** `python src/app.py --source 0 --line 100 400 1000 400 --csv metrics.csv`
- **Problemas comuns:**
  - Baixo FPS → reduzir resolução/fps, usar GPU, baixar `--max_fps`.
  - Webcam não abre → ver índice (`--source 1` etc.) ou permissões do SO.
  - Janela preta em servidor headless → usar arquivo de vídeo ou gravar saída em MP4 (feature futura).

---

## 12) Backlog Futuro (opcional)
- API REST `/metrics` + `/health` (FastAPI).
- Escrita de vídeo anotado (`--save out.mp4`).
- Multi‑zona e multi‑fonte simultâneas.
- Suporte a RTSP estável e reconexão.
- Exportar métricas para Prometheus/InfluxDB.
- Treinável para outros objetos (fila de carros, etc.).

---

## 13) Anexos — Modelos de Prompt (Gemini CLI & Codex)

### 13.1 Prompt Base (coloque antes de cada tarefa)
> “Você é um engenheiro sênior. **Siga estritamente meu plano .md**. Use **Python 3.10+**, escreva **código completo e pronto para rodar**, com **type hints** e **docstrings**. **Não omita arquivos** solicitados. Inclua testes `pytest` quando eu pedir. Comente decisões de projeto no topo de cada arquivo. Respeite os **requisitos** (RF/RNF) e **critérios de aceite** desta tarefa.”

### 13.2 Exemplo — Sprint 3 (Zonas) — Prompt de Pedido de Código
> “Com base no plano .md, implemente `zones.py` e altere `app.py` para suportar **linha** e **ROI** conforme RF-04 e RF-05. Entradas: `--line x1 y1 x2 y2`, `--roi x y ...`. Saídas: contagens e JSON por segundo. Gere os arquivos completos, incluindo qualquer utilitário necessário.”

### 13.3 Exemplo — Testes — Prompt
> “Implemente testes `pytest` para `zones.py` (interseção de linha e presença em polígono) e para `metrics.py` (CSV com cabeçalho e append). Forneça cenários positivos e negativos.”

### 13.4 Exemplo — Revisão entre modelos
> “Agora **revise** o código gerado anteriormente e aponte otimizações de performance e clareza. Sugira melhorias rápidas sem alterar a API pública.”

---

## 14) Definição de Pronto (DoD) — Geral
- Código executa em uma máquina de desenvolvimento comum **sem erros**.
- Funcionalidades do MVP (RF-01 a RF-10) verificadas.
- Testes automatizados cobrindo as partes críticas (zonas/CSV) passando.
- README e comentários suficientes para outro dev continuar.

---

## 15) Riscos & Mitigações
- **Acurácia em cenas difíceis** → Linha/ROI bem posicionadas; ajustar `conf/iou`; avaliar câmera/ângulo.
- **Desempenho baixo em CPU** → reduzir resolução/fps; usar `yolov8n`; desativar trilhas longas.
- **Licenciamentos** → revisar AGPL-3 se for distribuir binários fechados; planejar alternativa caso necessário.

---

## 16) Cronograma sugerido (estimativa)
- **Dia 1:** Sprints 0–2 (bootstrap, detecção, tracking).  
- **Dia 2:** Sprints 3–5 (zonas, visualização, telemetria).  
- **Dia 3:** Sprint 6–8 (testes, tuning, docs/empacote).

---

> **Conclusão:** Este plano é suficiente para que *Gemini CLI* e *Codex* entreguem o MVP completo, com qualidade e testes. Use os **prompts modelo** por sprint e cobre **arquivos completos** sempre que solicitar alterações.
