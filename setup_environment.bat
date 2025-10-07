@echo off
REM Скрипт для настройки окружения для лабораторных работ по компьютерному зрению (Windows)

echo 🚀 Настройка окружения для лабораторных работ...
echo.

REM Проверка версии Python
echo 1️⃣ Проверка версии Python...
python --version
if errorlevel 1 (
    echo    ❌ Python не найден. Установите Python 3.8 или выше.
    pause
    exit /b 1
)
echo.

REM Создание виртуального окружения
echo 2️⃣ Создание виртуального окружения...
if exist venv (
    echo    ⚠️  Виртуальное окружение уже существует
    set /p recreate="   Хотите пересоздать его? (y/n): "
    if /i "%recreate%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo    ✓ Виртуальное окружение пересоздано
    )
) else (
    python -m venv venv
    echo    ✓ Виртуальное окружение создано
)
echo.

REM Активация виртуального окружения
echo 3️⃣ Активация виртуального окружения...
call venv\Scripts\activate.bat
echo    ✓ Окружение активировано
echo.

REM Обновление pip
echo 4️⃣ Обновление pip...
python -m pip install --upgrade pip --quiet
echo    ✓ pip обновлён
echo.

REM Установка зависимостей
echo 5️⃣ Установка зависимостей из requirements.txt...
pip install -r requirements.txt --quiet
echo    ✓ Все зависимости установлены
echo.

REM Регистрация kernel для Jupyter
echo 6️⃣ Регистрация kernel для Jupyter Notebook...
python -m ipykernel install --user --name=cv-lab-env --display-name="Python (CV Labs)"
echo    ✓ Kernel зарегистрирован
echo.

REM Создание папки для тестовых изображений
echo 7️⃣ Создание папки для тестовых изображений...
if not exist test_images mkdir test_images
echo    ✓ Папка test_images создана
echo.

REM Итоговое сообщение
echo ✅ Настройка завершена успешно!
echo.
echo 📝 Следующие шаги:
echo    1. Активируйте виртуальное окружение: venv\Scripts\activate
echo    2. Поместите тестовые изображения в папку test_images\
echo    3. Запустите GUI приложение: python lab1_app.py или python lab2_app.py
echo    4. Или запустите Jupyter: jupyter notebook
echo.
echo 💡 В Jupyter выберите kernel: Kernel ^> Change Kernel ^> Python (CV Labs)
echo.
pause

