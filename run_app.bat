@echo off
title Chatbot AI Server

:loop
echo Pornire server Streamlit...
python -m streamlit run app.py

echo.
echo [ATENTIE] Aplicatia s-a oprit sau a intampinat o eroare fatala!
echo Se va reporni automat in 5 secunde...
timeout /t 5 /nobreak > NUL
goto loop
