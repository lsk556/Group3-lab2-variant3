@echo off
echo Running mypy...
mypy binary_tree.py test_binary_tree.py
if errorlevel 1 exit /b %errorlevel%

echo Running pycodestyle...
pycodestyle binary_tree.py test_binary_tree.py
if errorlevel 1 exit /b %errorlevel%

echo Running pyflakes...
pyflakes binary_tree.py test_binary_tree.py
if errorlevel 1 exit /b %errorlevel%

echo Running tests with coverage...
pytest --cov=binary_tree test_binary_tree.py
if errorlevel 1 exit /b %errorlevel%

echo All checks passed!