@echo off
set PYTHON_EXE=C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe
echo Starting report generation at %TIME% > report_log.txt
"%PYTHON_EXE%" generate_report.py >> report_log.txt 2>&1
echo Report generation done. >> report_log.txt
type report_log.txt
