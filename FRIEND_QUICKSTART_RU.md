# Быстрый запуск на ПК с RTX 5070/50-series

Нужна 64-битная Windows 10/11, GeForce RTX 5070 или другая Blackwell RTX 50-series и
желательно не менее 25 GB свободного места.
Python, PyTorch, CUDA Toolkit и cuDNN на Windows отдельно устанавливать не нужно.

1. Скачайте ZIP ветки `main` и распакуйте его либо выполните:

   ```text
   git clone --branch main https://github.com/maggogerka/fly-abstraction-lab.git
   cd fly-abstraction-lab
   ```

2. Дважды щёлкните `START_HERE.cmd`. Скрипт проверит Windows x64, место, драйвер NVIDIA,
   WSL2, Git, Docker Desktop/Compose/engine, соберёт закреплённый образ и выполнит GPU
   doctor и smoke. Логи останутся в `results/diagnostics/`.

   Если старая копия из ZIP выводит ошибку вида `'shell.exe' is not recognized`, удалите
   её и скачайте свежий ZIP `main`. Для немедленного запуска без CMD-обёртки откройте
   Windows PowerShell в корне проекта и выполните:

   ```text
   powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_friend_pc.ps1" -Action Guided
   ```

   Если старая версия пишет `nvidia-smi найден, но не работает`, но перед этим уже
   показывает RTX 5070, драйвер и VRAM, это ошибка старого скрипта, а не видеокарты.
   Удалите старую распакованную папку и скачайте свежий ZIP ветки `main`.

   Если старая версия останавливается на строке `Image ... Building`, Docker не обязательно
   сломан: Docker Desktop пишет обычный прогресс сборки в stderr, а старый PowerShell-скрипт
   принимал его за ошибку. В свежем ZIP это исправлено. Первый образ большой (примерно 4 GB),
   поэтому не закрывайте окно и дождитесь окончания загрузки.

   Если старая версия после установки пакетов завершилась на `AssertionError` в
   `Dockerfile.gpu`, скачайте свежий ZIP `main`. Раньше build-time проверка ошибочно
   запрашивала видимую видеокарту, хотя Docker не предоставляет GPU во время сборки.

   Ошибка `index_copy_(): self and source expected to have the same dtype` в старом
   `smoke-gpu` также исправлена в свежем ZIP. Она относилась к BF16 autocast модели,
   а не к RTX 5070 или драйверу.

3. Если нет Git или Docker Desktop, установка через `winget` начнётся только после
   точной фразы, показанной скриптом. Драйвер NVIDIA устанавливайте только вручную с
   официального сайта. После WSL/Docker/драйвера перезагрузите Windows и повторите шаг 2.

4. В меню сначала выберите сведения об UCI. Для отдельной загрузки выберите пункт 6 и
   введите `DOWNLOAD UCI`, затем пункт 7 для подготовки.

5. Выберите пункт 8. Это `pilot_gpu` dry-run: он только показывает конфигурацию и оценку
   RAM/VRAM, не обучает модель и не создаёт run.

6. Только после проверки dry-run выберите пункт 9 и введите точную фразу `TRAIN PILOT`.
   Закрытие окна не удаляет `data/` и `results/`; повторный запуск setup безопасен.

7. После пилота отправьте владельцу:

   - всю папку `results/diagnostics/`;
   - SHA commit из `git rev-parse HEAD` (если использовался Git);
   - `data/raw/uci_energy_efficiency/data.csv.download.json`;
   - `data/processed/uci_energy_efficiency.jsonl.manifest.json`;
   - использованный JSON из `data/processed/splits/`;
   - из `results/<pilot_run_id>/`: `config.resolved.yaml`, `run_manifest.json`,
     `summary.json`, `metrics.json`, `history.csv`, `predictions.jsonl`;
   - полный `.log` пилота из `results/diagnostics/`, особенно если произошла ошибка.

`checkpoint.pt` отправляйте только если нужно продолжить обучение. Не отправляйте
FlyWire-экспорты без отдельного разрешения на распространение.

## FlyWire FAFB v783: отдельный безопасный сценарий

Используется только публичный статический файл
`proofread_connections_783.feather` из Zenodo, DOI
`10.5281/zenodo.10676866`. Размер файла — 852 022 274 байта, опубликованный MD5 —
`f48f972d262323a102aed49af1396b8a`. Лицензия в карточке набора не указана, поэтому
перед загрузкой проверьте карточку и условия Zenodo.

В меню действия разделены:

- 10 — только показать метаданные;
- 11 — загрузить после точной фразы `DOWNLOAD FLYWIRE V783`;
- 12/14 — dry-run подготовки 512/1024 без создания файлов;
- 13/15 — подготовить после `PREPARE FLYWIRE 512` или `PREPARE FLYWIRE 1024`;
- 16/17 — проверить готовый граф;
- 18 — dry-run обучения без создания run;
- 19 — одна smoke-эпоха только после `TRAIN FLYWIRE SMOKE`.

Для RTX 5070 с примерно 12 GiB VRAM сначала используйте 512 узлов, затем 1024. К 2048
переходите только после проверки логов RAM/VRAM; 4096 без этих измерений не запускайте.
Отображения входных и выходных узлов искусственные и независимые от задачи — это не
сенсорные и не моторные нейроны.

После FlyWire smoke отправьте владельцу:

- `results/diagnostics/flywire-*.log` и полный лог запуска;
- `proofread_connections_783.feather.download.json`;
- `flywire_v783_core_512.manifest.json`, `.stats.json` и `.nodes.csv`;
- из каталога run: `config.resolved.yaml`, `run_manifest.json`, `summary.json`,
  `metrics.json`, `history.csv` и `predictions.jsonl`.

Сам Feather/NPZ отправляйте только если условия источника разрешают распространение.
