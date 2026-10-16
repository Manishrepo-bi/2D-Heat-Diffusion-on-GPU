@echo off
rem Starts the local GitHub emulator (Vercel Labs "emulate") on http://127.0.0.1:4000
rem Keep this window open while the team works. Ctrl+C stops it.
cd /d "%~dp0"
npx --yes emulate@0.12.2 start --service github --seed emulate.config.json %*
