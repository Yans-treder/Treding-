#!/usr/bin/env python3
"""
Bot de Trading con IA para Replit
Ejecuta Freqtrade con estrategia de IA
"""
import os
import subprocess
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def install_dependencies():
    """Instalar dependencias de Freqtrade"""
    print("Instalando dependencias...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_freqtrade():
    """Configurar entorno de Freqtrade"""
    print("Configurando Freqtrade...")
    
    # Crear estructura de directorios
    os.makedirs("user_data/strategies", exist_ok=True)
    os.makedirs("user_data/data", exist_ok=True)
    
    # Verificar que la estrategia existe
    if not os.path.exists("user_data/strategies/AdvancedAIStrategy.py"):
        print("ERROR: No se encuentra AdvancedAIStrategy.py")
        sys.exit(1)

def download_data():
    """Descargar datos históricos"""
    print("Descargando datos históricos...")
    result = subprocess.run([
        sys.executable, "-m", "freqtrade", "download-data",
        "--exchange", "binance",
        "--pairs", "BTC/USDT", "ETH/USDT",
        "--timeframes", "15m",
        "--days", "60"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

def train_ai_model():
    """Entrenar modelo de IA"""
    print("Entrenando modelo de IA...")
    try:
        from train_model import main as train_main
        train_main()
    except Exception as e:
        print(f"Error entrenando modelo: {e}")

def run_backtest():
    """Ejecutar backtesting"""
    print("Ejecutando backtesting...")
    result = subprocess.run([
        sys.executable, "-m", "freqtrade", "backtesting",
        "--config", "config.json",
        "--strategy", "AdvancedAIStrategy",
        "--timeframe", "15m",
        "--export", "signals"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

def main():
    """Función principal"""
    print("=== Bot de Trading con IA en Replit ===")
    
    # Instalar dependencias
    install_dependencies()
    
    # Configurar Freqtrade
    setup_freqtrade()
    
    # Descargar datos
    download_data()
    
    # Entrenar modelo (opcional)
    if input("¿Entrenar modelo de IA? (s/n): ").lower() == 's':
        train_ai_model()
    
    # Ejecutar backtest
    if input("¿Ejecutar backtesting? (s/n): ").lower() == 's':
        run_backtest()
    
    print("Proceso completado. Revisa los resultados.")

if __name__ == "__main__":
    main()
