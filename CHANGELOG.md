# Changelog

## [Sprint 9] - 2025-10-02

### ✨ Novas Funcionalidades

#### 1. Argumentos CLI --line e --roi Implementados
- **`--line x1 y1 x2 y2`**: Permite configurar linha personalizada para contagem de cruzamentos
- **`--roi x1 y1 x2 y2 ...`**: Permite configurar polígono ROI personalizado (mínimo 3 pontos)
- Se não especificados, usa valores padrão (linha horizontal no meio / ROI retangular centralizado)

#### 2. Modo Headless
- **`--headless`**: Executa aplicação sem interface gráfica (cv2.imshow)
- **`--output-video <arquivo>`**: Salva vídeo anotado em arquivo
- Útil para processamento em servidores sem GUI

#### 3. Validações Aprimoradas
- **Validação de modelo**: Verifica se arquivo existe ou é modelo YOLOv8 válido antes de inicializar
- **Validação de coordenadas**: Verifica se linha/ROI estão dentro dos limites do frame
- **Validação de argumentos**: Verifica dependências (headless requer output-video, ROI requer mínimo 3 pontos)

### 🧪 Testes
- Novos testes para validação de modelo (`test_app_validation.py`)
- Teste de pipeline com linha/ROI personalizados
- Suporte a VideoWriter no conftest.py

### 📚 Documentação
- README atualizado com novos argumentos e exemplos
- CLAUDE.md atualizado com informações sobre novas features
- Documentação bilíngue (PT-BR e EN)

### 🔧 Melhorias Técnicas
- Código formatado com Black (88 caracteres)
- Linting aprovado com Ruff
- 16 testes passando (100% de sucesso)
- Type hints mantidos em todas as funções

### 📝 Arquivos Modificados
- `src/app.py`: +80 linhas (validações, argumentos CLI, modo headless)
- `tests/test_pipeline_smoke.py`: teste adicional com linha/ROI personalizados
- `tests/test_app_validation.py`: novo arquivo com 3 testes
- `tests/conftest.py`: suporte a VideoWriter e CAP_PROP_FPS
- `README.md`: documentação atualizada
- `CLAUDE.md`: guia atualizado
- `CHANGELOG.md`: este arquivo

### 🐛 Correções
- **Problema crítico resolvido**: argumentos `--line` e `--roi` estavam documentados mas não implementados
- Formatação de código (linhas > 88 caracteres)
- Imports não utilizados removidos

---

## Commits Anteriores

### [Sprint 7 & 8] - 2025-09-08
- Melhorias gerais no pipeline

### [Sprint 6] - 2025-09-08
- Testes adicionados

### [Sprint 5] - 2025-09-08
- Desenha linha A–B tracejada, ROI, bboxes e centro + legenda no vídeo

### [Sprint 3] - 2025-09-08
- Correção de blocos de import e integração LineCounter/RoiCounter no app

### [Sprint 2] - 2025-09-08
- Integração do MultiObjectTracker

### [Sprint 1] - 2025-09-08
- Integração do PersonDetector e logging de métricas

### [Initial] - 2025-09-08
- Estrutura inicial do projeto
