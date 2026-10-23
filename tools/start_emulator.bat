@echo off
cd /d "%~dp0"
npx --yes emulate@0.12.2 start --service github --seed emulate.config.json %*
