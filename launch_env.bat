@echo off
title Hermetic Sandbox Environment
color 0b
set "ROOT_DIR=%~dp0"
set "PYTHONPATH="
set "PYTHONCASEOK="
set "VIRTUAL_ENV="
set "PYTHONIOENCODING=utf-8"
set "PYTHONHOME=%ROOT_DIR%py_env_3_14_7"
set "PATH=%ROOT_DIR%py_env_3_14_7;%ROOT_DIR%py_env_3_14_7\Scripts;%PATH%"
set "HF_HOME=%ROOT_DIR%.hf_cache"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo =======================================================================
echo  Hermetic Sandbox Shell Active
echo  Interpreter: %PYTHONHOME%\python.exe
echo  Cache Anchor: %HF_HOME%
echo  Isolation Status: Host PYTHONPATH cleared. Anchored relative root.
echo =======================================================================
cmd /k
