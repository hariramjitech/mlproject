@echo off
set PYTHON_EXE=C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe
echo Starting job at %TIME% > job_log.txt
echo Using Python at %PYTHON_EXE% >> job_log.txt
echo Running evaluation... >> job_log.txt
"%PYTHON_EXE%" model_evaluation.py >> job_log.txt 2>&1
echo Evaluation done. >> job_log.txt
type job_log.txt
