@echo off
rem Builds dist\modelview.exe (one file, no console window).
rem Needs Python 3.9+ with:  pip install -r requirements.txt pyinstaller
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name modelview ^
  --icon "%~dp0modelview\icon.ico" ^
  --add-data "%~dp0modelview\icon.ico;modelview" --add-data "%~dp0modelview\icon.png;modelview" ^
  --collect-all moderngl --collect-all glcontext --collect-all pyglet ^
  --workpath "%TEMP%\modelview-build" --specpath "%TEMP%\modelview-build" --distpath "%~dp0dist" app.py
if errorlevel 1 exit /b 1
echo.
echo Built dist\modelview.exe
