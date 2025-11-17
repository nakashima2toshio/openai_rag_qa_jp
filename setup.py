#!/usr/bin/env python3
"""
MCP環境セットアップスクリプト
Usage: python setup.py [--quick]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Python バージョンチェック"""
    print("🐍 Python バージョンチェック...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8以上が必要です。現在のバージョン: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True


def check_qdrant_server():
    """Qdrantサーバー接続確認"""
    print("🔍 Qdrantサーバー接続確認...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()
        print("✅ Qdrantサーバーが稼働中")
        return True
    except Exception as e:
        print(f"⚠️ Qdrantサーバーに接続できません: {e}")
        return False


def start_qdrant_docker():
    """DockerでQdrantを起動"""
    print("🐳 Qdrantサーバーを起動中...")
    try:
        # Docker Composeで起動
        docker_compose_path = Path("docker-compose/docker-compose.yml")
        if docker_compose_path.exists():
            subprocess.run(["docker-compose", "-f", str(docker_compose_path), "up", "-d", "qdrant"], check=True)
            print("✅ Qdrantがdocker-composeで起動しました")
        else:
            # 単独でDocker起動
            subprocess.run([
                "docker", "run", "-d", 
                "--name", "qdrant", 
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant"
            ], check=True)
            print("✅ QdrantがDockerで起動しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Qdrant起動失敗: {e}")
        print("手動でQdrantを起動してください:")
        print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False


def detect_package_manager():
    """パッケージマネージャーの検出"""
    if subprocess.run(["which", "uv"], capture_output=True).returncode == 0:
        print("✅ uvを使用してパッケージをインストール")
        return "uv"
    else:
        print("📦 pipを使用してパッケージをインストール")
        return "pip"


def install_packages(package_manager="pip"):
    """必要なパッケージをインストール"""
    print("📦 必要なパッケージをインストール中...")
    
    if package_manager == "uv":
        packages = [
            "streamlit>=1.48.0",
            "openai>=1.99.9", 
            "fastapi>=0.116.1",
            "uvicorn[standard]>=0.24.0",
            "psycopg2-binary>=2.9.0",
            "qdrant-client>=1.6.0",
            "pandas>=2.0.0",
            "requests>=2.31.0",
            "python-dotenv>=1.0.0"
        ]
        
        try:
            # uvプロジェクトを初期化（既存の場合はスキップ）
            if not Path("pyproject.toml").exists():
                subprocess.run(["uv", "init", "."], check=True)
            
            for package in packages:
                subprocess.run(["uv", "add", package], check=True)
            print("✅ uvでのインストール完了")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ uvでのインストール失敗: {e}")
            return False
    else:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
            ], check=True, capture_output=True)
            print("✅ pipでのインストール完了")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ pipでのインストール失敗: {e}")
            return False


def create_env_template():
    """環境変数テンプレートファイルの作成"""
    print("🔧 .env テンプレートファイル作成中...")
    
    env_template = """# OpenAI API Key (必須)
OPENAI_API_KEY=your-openai-api-key-here

# Database URLs
PG_CONN_STR=postgresql://testuser:testpass@localhost:5432/testdb
QDRANT_URL=http://localhost:6333

# Optional API Keys
PINECONE_API_KEY=your-pinecone-api-key-here
"""
    
    if not Path(".env").exists():
        Path(".env").write_text(env_template)
        print("⚠️  .env ファイルを作成しました。OPENAI_API_KEY を設定してください。")
    else:
        print("✅ .env ファイルは既に存在します")


def verify_installation():
    """インストール確認"""
    print("🔍 インストール確認中...")
    
    required_modules = [
        "streamlit", "openai", "fastapi", "uvicorn", 
        "psycopg2", "qdrant_client",
        "pandas", "requests", "yaml", "datasets"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == "yaml":
                __import__("yaml")
            elif module == "datasets":
                __import__("datasets")
            else:
                __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ 不足パッケージ: {', '.join(missing_modules)}")
        return False
    
    print("✅ 全ての依存関係が満たされています")
    return True


def setup_qdrant_data():
    """Qdrantにデータを登録"""
    print("📊 Qdrantにデータを登録中...")
    try:
        # a50_qdrant_registration.pyが存在するか確認
        if Path("a50_qdrant_registration.py").exists():
            subprocess.run([sys.executable, "a42_qdrant_registration.py", "--recreate", "--limit", "100"], check=True)
            print("✅ Qdrantへのデータ登録完了")
            return True
        else:
            print("⚠️ a50_qdrant_registration.pyが見つかりません")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ データ登録失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MCP環境セットアップ")
    parser.add_argument("--quick", action="store_true", help="クイックセットアップ（検証をスキップ）")
    args = parser.parse_args()
    
    print("🚀 MCP環境セットアップを開始します")
    print("=" * 50)
    
    # 1. Python バージョンチェック
    if not check_python_version():
        sys.exit(1)
    
    # 2. パッケージマネージャー検出
    package_manager = detect_package_manager()
    
    # 3. パッケージインストール
    if not install_packages(package_manager):
        print("❌ セットアップ失敗: パッケージのインストールに失敗")
        sys.exit(1)
    
    # 4. 環境変数テンプレート作成
    create_env_template()
    
    # 5. インストール確認（クイックモードでない場合）
    if not args.quick:
        if not verify_installation():
            print("❌ セットアップ失敗: 一部のパッケージが不足")
            sys.exit(1)
    
    # 6. Qdrantサーバーの確認と起動
    if not check_qdrant_server():
        print("\n🐳 Qdrantサーバーを起動します...")
        start_qdrant_docker()
        import time
        time.sleep(5)  # 起動を待つ
        if not check_qdrant_server():
            print("⚠️ Qdrantサーバーの自動起動に失敗しました")
    
    # 7. Qdrantデータ登録（オプション）
    if check_qdrant_server():
        response = input("\nQdrantにサンプルデータを登録しますか？ (y/n): ")
        if response.lower() == 'y':
            setup_qdrant_data()
    
    print("\n🎉 環境セットアップ完了!")
    print("\n次のステップ:")
    print("1. .env ファイルで OPENAI_API_KEY を設定")
    print("2. Qdrantサーバー起動: docker-compose -f docker-compose/docker-compose.yml up -d")
    print("3. データ登録: python a42_qdrant_registration.py --recreate")
    print("4. サーバー起動: python server.py")
    print("5. 検索UI起動: streamlit run a50_rag_search_local_qdrant.py")


if __name__ == "__main__":
    main()