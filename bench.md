# Análise de Performance (Benchmark)

Este documento registra os resultados dos testes de performance da aplicação de contagem de pessoas, comparando diferentes modelos e parâmetros.

**Ambiente de Teste:**
- **CPU:** (a ser preenchido)
- **GPU:** (a ser preenchido)
- **RAM:** (a ser preenchido)
- **Vídeo de Teste:** (a ser preenchido, ex: 720p, 30fps, 60s)

## Tabela de Resultados

| Modelo    | `--conf` | `--iou` | FPS Médio (Processamento) |
| :-------- | :------- | :------ | :------------------------ |
| `yolov8n.pt` | 0.35     | 0.45    | (a ser preenchido)        |
| `yolov8n.pt` | 0.50     | 0.45    | (a ser preenchido)        |
| `yolov8s.pt` | 0.35     | 0.45    | (a ser preenchido)        |
| `yolov8s.pt` | 0.50     | 0.45    | (a ser preenchido)        |

---

## Análise e Recomendações

*(Esta seção será preenchida após a coleta dos resultados.)*

### Trade-offs: `yolov8n` vs. `yolov8s`

- **yolov8n (nano):**
  - *Prós:* Muito leve e rápido, ideal para sistemas com CPU menos potente ou ambientes embarcados.
  - *Contras:* Menor precisão, pode ter mais dificuldade em detectar pessoas em cenas complexas, com oclusão ou à distância.

- **yolov8s (small):**
  - *Prós:* Melhor precisão que o modelo nano, oferecendo um bom equilíbrio entre performance e acurácia.
  - *Contras:* Mais pesado e lento que o nano, exigindo mais recursos de CPU/GPU.

### Recomendações de Parâmetros Padrão

- **`--conf` (Confidence Threshold):**
  - Um valor inicial de `0.35` é um bom ponto de partida. Aumentar para `0.50` ou mais pode reduzir falsos positivos, mas com o risco de perder detecções válidas.

- **`--iou` (Intersection over Union):**
  - O valor de `0.45` é um padrão razoável para evitar caixas de detecção duplicadas na mesma pessoa.
