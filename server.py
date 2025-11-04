#!/usr/bin/env python3
"""
Qdrantサーバー起動スクリプト  
Usage: python server.py [--port PORT] [--test]
"""

import os
import sys
import time
import subprocess
import argparse
import requests
from pathlib import Path


def check_qdrant_connection():
    """Qdrant接続確認"""
    connections_ok = True
    
    # Qdrant接続確認
    print("🔍 Qdrant接続確認...")
    try:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        client = QdrantClient(url=qdrant_url, timeout=5)
        collections = client.get_collections()
        print(f"  ✅ Qdrant接続成功 (コレクション数: {len(collections.collections)})")
    except ImportError:
        print("  ❌ qdrant-client パッケージが不足しています")
        connections_ok = False
    except Exception as e:
        print(f"  ❌ Qdrant接続失敗: {e}")
        connections_ok = False
    
    if not connections_ok:
        print("\n💡 解決方法:")
        print("1. Qdrantサーバーを起動:")
        print("   cd docker-compose && docker-compose up -d")
        print("2. Qdrantデータを投入:")
        print("   python qdrant_data_loader.py --recreate")
    
    return connections_ok


def start_qdrant_server():
    """Qdrantサーバーを起動"""
    print("🐳 Qdrantサーバーを確認中...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()
        print("✅ Qdrantサーバーは既に稼働中")
        return True
    except Exception:
        print("🐳 QdrantサーバーをDocker Composeで起動中...")
        try:
            # Docker Composeを優先
            docker_compose_path = Path("docker-compose/docker-compose.yml")
            if docker_compose_path.exists():
                subprocess.run([
                    "docker-compose", "-f", str(docker_compose_path), 
                    "up", "-d", "qdrant"
                ], check=True, capture_output=True)
            else:
                print("⚠️ docker-compose/docker-compose.yml が見つからないため、自動起動をスキップします")
            
            # 起動待機
            import time
            for _ in range(10):
                try:
                    client = QdrantClient(url="http://localhost:6333", timeout=5)
                    client.get_collections()
                    print("✅ Qdrantサーバーが起動しました")
                    return True
                except:
                    time.sleep(1)
            
            print("❌ Qdrantサーバーの起動に失敗（Docker Compose）")
            return False
        except Exception as e:
            print(f"❌ DockerによるQdrant起動失敗: {e}")
            return False

def start_api_server(port=8000):
    """APIサーバーを起動"""
    print(f"🚀 APIサーバーを起動中（ポート: {port}）...")
    
    # サーバーファイルの存在確認
    if not Path("mcp_api_server.py").exists():
        print("⚠️ mcp_api_server.py が見つかりません")
        print("🔍 代わりにQdrant検索UIを起動可能です")
        return None
    
    # バックグラウンドでAPIサーバーを起動
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "mcp_api_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # サーバーの起動を待機
        print("⏳ サーバーの起動を待機中...")
        for i in range(30):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    health_data = response.json()
                    print("✅ APIサーバーが起動しました!")
                    print(f"📍 URL: http://localhost:{port}")
                    print(f"📖 ドキュメント: http://localhost:{port}/docs")
                    print(f"🏥 ヘルス: {health_data.get('status', 'unknown')}")
                    return process
            except Exception:
                pass
            
            print(f"   ... 待機中 ({i + 1}/30)")
            time.sleep(1)
        
        print("❌ APIサーバーの起動に失敗しました")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ APIサーバー起動エラー: {e}")
        return None


def test_api_endpoints(port=8000):
    """APIエンドポイントのテスト"""
    print("🧪 APIエンドポイントのテスト...")
    
    base_url = f"http://localhost:{port}"
    test_endpoints = [
        ("GET", "/health", "ヘルスチェック"),
        ("GET", "/api/customers?limit=1", "顧客一覧"),
        ("GET", "/api/products?limit=1", "商品一覧"),
        ("GET", "/api/stats/sales", "売上統計")
    ]
    
    successful_tests = 0
    for method, endpoint, description in test_endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.request(method, url, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ {description}: OK")
                successful_tests += 1
            else:
                print(f"  ⚠️ {description}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ {description}: エラー - {e}")
    
    print(f"\n📊 テスト結果: {successful_tests}/{len(test_endpoints)} 成功")
    return successful_tests == len(test_endpoints)


def display_usage_info(port=8000):
    """使用方法の表示"""
    print("\n" + "=" * 50)
    print("📚 使用方法")
    print("=" * 50)
    
    print(f"\n💡 基本的な使用方法:")
    print(f"🌐 APIドキュメント: http://localhost:{port}/docs")
    print(f"📖 ReDoc: http://localhost:{port}/redoc")
    print(f"🏥 ヘルスチェック: curl http://localhost:{port}/health")
    
    print(f"\n🔧 サーバー管理:")
    print("- サーバー停止: Ctrl+C")
    print(f"- ポート確認: netstat -an | grep {port}")


def start_streamlit_ui():
    """ストリームリットUIを起動"""
    print("🌐 Streamlit UIを起動中...")
    if Path("a50_qdrant_search.py").exists():
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "a50_rag_search_local_qdrant.py",
                "--server.port", "8504"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ Streamlit UIが起動しました")
            print("📄 URL: http://localhost:8504")
            return process
        except Exception as e:
            print(f"❌ Streamlit UI起動失敗: {e}")
            return None
    else:
        print("❌ a50_rag_search_local_qdrant.py が見つかりません")
        return None

def main():
    parser = argparse.ArgumentParser(description="Qdrantサーバー起動")
    parser.add_argument("--port", type=int, default=8000, help="APIポート番号（デフォルト: 8000）")
    parser.add_argument("--test", action="store_true", help="起動後にテストを実行")
    parser.add_argument("--no-ui", action="store_true", help="Streamlit UIを起動しない")
    args = parser.parse_args()
    
    print("🚀 Qdrantサーバー起動を開始します")
    print("=" * 50)
    
    # 0. Qdrantサーバー起動
    if not start_qdrant_server():
        print("❌ Qdrantサーバーの起動に失敗しました")
        print("手動で起動してください:")
        print("  cd docker-compose && docker-compose up -d")
        sys.exit(1)
    
    # 1. Qdrant接続確認
    if not check_qdrant_connection():
        print("❌ Qdrant接続に失敗しました")
        print("データを投入してください: python qdrant_data_loader.py --recreate")
        sys.exit(1)
    
    # 2. APIサーバー起動（オプション）
    server_process = start_api_server(args.port)
    
    # 3. Streamlit UI起動（オプション）
    ui_process = None
    if not args.no_ui:
        ui_process = start_streamlit_ui()
    
    # 3. テスト実行（指定された場合）
    if args.test:
        if test_api_endpoints(args.port):
            print("✅ 全てのエンドポイントが正常です")
        else:
            print("⚠️ 一部のエンドポイントに問題があります")
    
    # 4. 使用方法表示
    display_usage_info(args.port)
    
    print("\n🎉 サーバー起動完了!")
    print("\n📚 利用可能なコマンド:")
    print("  データ登録: python qdrant_data_loader.py --recreate")
    print("  詳細登録: python a42_qdrant_registration.py --recreate")
    if ui_process:
        print("  検索UI: http://localhost:8504")
    print("\n⏸️ サーバーを停止するには Ctrl+C を押してください...")
    
    try:
        if server_process:
            server_process.wait()
        elif ui_process:
            ui_process.wait()
        else:
            # どちらも起動していない場合はキー入力待ち
            input("\nEnterキーで終了")
    except KeyboardInterrupt:
        print("\n🛑 サーバーを停止中...")
        if server_process:
            server_process.terminate()
            server_process.wait()
        if ui_process:
            ui_process.terminate()
            ui_process.wait()
        print("✅ サーバーが停止しました")


if __name__ == "__main__":
    main()
