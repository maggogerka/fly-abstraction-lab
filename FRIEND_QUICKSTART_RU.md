# Быстрый запуск на ПК с RTX 5070/50-series

Нужна 64-битная Windows 10/11, GeForce RTX 5070 или другая Blackwell RTX 50-series и
желательно не менее 25 GB свободного места.
Python, PyTorch, CUDA Toolkit и cuDNN на Windows отдельно устанавливать не нужно.

1. Скачайте ZIP ветки `feat/research-pc-readiness` и распакуйте его либо выполните:

   ```text
   git clone --branch feat/research-pc-readiness https://github.com/maggogerka/fly-abstraction-lab.git
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
