#!/bin/bash
# Скрипт для настройки окружения для лабораторных работ по компьютерному зрению

echo "🚀 Настройка окружения для лабораторных работ..."
echo ""

# Проверка версии Python
echo "1️⃣ Проверка версии Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Найден Python: $python_version"

# Создание виртуального окружения
echo ""
echo "2️⃣ Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo "   ⚠️  Виртуальное окружение уже существует"
    read -p "   Хотите пересоздать его? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        rm -rf venv
        python3 -m venv venv
        echo "   ✓ Виртуальное окружение пересоздано"
    fi
else
    python3 -m venv venv
    echo "   ✓ Виртуальное окружение создано"
fi

# Активация виртуального окружения
echo ""
echo "3️⃣ Активация виртуального окружения..."
source venv/bin/activate
echo "   ✓ Окружение активировано"

# Обновление pip
echo ""
echo "4️⃣ Обновление pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip обновлён"

# Установка зависимостей
echo ""
echo "5️⃣ Установка зависимостей из requirements.txt..."
pip install -r requirements.txt --quiet
echo "   ✓ Все зависимости установлены"

# Регистрация kernel для Jupyter
echo ""
echo "6️⃣ Регистрация kernel для Jupyter Notebook..."
python -m ipykernel install --user --name=cv-lab-env --display-name="Python (CV Labs)"
echo "   ✓ Kernel зарегистрирован"

# Создание папки для тестовых изображений
echo ""
echo "7️⃣ Создание папки для тестовых изображений..."
mkdir -p test_images
echo "   ✓ Папка test_images создана"

# Итоговое сообщение
echo ""
echo "✅ Настройка завершена успешно!"
echo ""
echo "📝 Следующие шаги:"
echo "   1. Активируйте виртуальное окружение: source venv/bin/activate"
echo "   2. Поместите тестовые изображения в папку test_images/"
echo "   3. Запустите GUI приложение: python lab1_app.py или python lab2_app.py"
echo "   4. Или запустите Jupyter: jupyter notebook"
echo ""
echo "💡 В Jupyter выберите kernel: Kernel > Change Kernel > Python (CV Labs)"
echo ""

