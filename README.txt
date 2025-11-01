
God-Eye Full Suite - DO BEST Edition
===================================
1) Extract package into a folder and open in Visual Studio Code.
2) Setup venv and install deps:
   python -m venv .venv
   & .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
3) Run run_all.ps1 (F5 in VS or double-click run_all.ps1)
4) To create a Desktop shortcut (optional):
   Run create_shortcut.ps1 from the project folder:
   powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1
Notes:
- If lightgbm fails to install, the system will fallback to the heuristic AI model; it will still run.
- Dashboard supports voice alerts via browser Web Speech API. Allow site audio.
