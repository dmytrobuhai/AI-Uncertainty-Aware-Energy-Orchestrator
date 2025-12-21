import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class APIConfigError(Exception):
    """Виняток для помилок конфігурації API"""
    pass


def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """
    Отримує змінну середовища з перевіркою наявності
    
    Args:
        key: Назва змінної середовища
        default: Значення за замовчуванням (якщо не required)
        required: Чи обов'язкова змінна
    
    Returns:
        Значення змінної середовища
    
    Raises:
        APIConfigError: Якщо required=True і змінна відсутня
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise APIConfigError(
            f"Змінна середовища '{key}' не знайдена. "
            f"Переконайтеся, що файл .env існує та містить {key}."
        )
    
    if required and value == default and default is None:
        raise APIConfigError(
            f"Змінна середовища '{key}' не встановлена. "
            f"Додайте її у файл .env"
        )
    
    return value


def get_openai_client(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None
) -> ChatOpenAI:
    """
    Створює клієнт OpenAI для використання в LangChain/LangGraph
    
    Args:
        model: Назва моделі (за замовчуванням з .env)
        temperature: Температура генерації (за замовчуванням з .env)
        base_url: Базовий URL API (за замовчуванням з .env або стандартний)
    
    Returns:
        Екземпляр ChatOpenAI
    
    Raises:
        APIConfigError: Якщо API ключ відсутній
    """
    api_key = get_env_var("OPENAI_API_KEY", required=True)
    
    # Отримуємо параметри з .env або використовуємо передані
    model_name = model or get_env_var("OPENAI_MODEL", default="gpt-4o-mini", required=False)
    temp = temperature if temperature is not None else float(
        get_env_var("OPENAI_TEMPERATURE", default="0.7", required=False)
    )
    base = base_url or get_env_var("OPENAI_BASE_URL", default=None, required=False)
    
    config = {
        "model": model_name,
        "temperature": temp,
        "api_key": api_key,
    }
    
    if base:
        config["base_url"] = base
    
    return ChatOpenAI(**config)


def get_model_config() -> dict:
    """
    Отримує конфігурацію моделі з .env файлу
    
    Returns:
        Словник з параметрами моделі
    """
    return {
        "model": get_env_var("OPENAI_MODEL", default="gpt-4o-mini", required=False),
        "temperature": float(
            get_env_var("OPENAI_TEMPERATURE", default="0.7", required=False)
        ),
        "base_url": get_env_var("OPENAI_BASE_URL", default=None, required=False),
    }


def check_api_keys() -> dict:
    """
    Перевіряє наявність всіх необхідних API ключів
    
    Returns:
        Словник зі статусом кожного ключа
    """
    keys_status = {}
    
    # Перевірка OpenAI ключа
    try:
        openai_key = get_env_var("OPENAI_API_KEY", required=True)
        keys_status["OPENAI_API_KEY"] = "✓ Встановлено" if openai_key else "✗ Відсутній"
    except APIConfigError:
        keys_status["OPENAI_API_KEY"] = "✗ Відсутній"
    
    # Перевірка опціональних ключів
    optional_keys = [
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "WEATHER_API_KEY",
        "ENERGY_SYSTEM_API_KEY"
    ]
    
    for key in optional_keys:
        value = get_env_var(key, required=False)
        keys_status[key] = "✓ Встановлено" if value else "○ Не встановлено (опціонально)"
    
    return keys_status


def print_config_status():
    """
    Виводить статус конфігурації API ключів
    """
    
    status = check_api_keys()
    for key, value in status.items():
        print(f"{key:30s}: {value}")
    
    print("\nКонфігурація моделі:")
    try:
        config = get_model_config()
        for key, value in config.items():
            if value:
                print(f"  {key:20s}: {value}")
    except Exception as e:
        print(f"  Помилка: {e}")
    



if __name__ == "__main__":
    print_config_status()

