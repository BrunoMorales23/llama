@echo on
set FILE_PATH=C:\Users\BOTSMS\Desktop\INPUTS\CARBON HOYTS ABASTO 1025.pdf

call "C:\Users\BOTSMS\Desktop\SCRIPTS\Script Llama\env\Scripts\python.exe" ^
    "%USERPROFILE%\Desktop\SCRIPTS\dyn\ocr.py" ^
    "%FILE_PATH%"
pause