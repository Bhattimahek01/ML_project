@echo off
echo Installing dependencies...
pip install flask scikit-learn numpy

echo Starting CardioGuard Application...
echo Open your browser to: http://127.0.0.1:5000
python main.py
pause
