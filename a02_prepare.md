 a02_make_qa.py 並列化提案とTodoリスト

  📋 現状分析とボトルネック

  現在の処理フロー:
  1. 逐次処理: 各チャンクを順番に処理（generate_qa_pairs_for_chunk）
  2. バッチ処理: 複数チャンクをまとめて処理（generate_qa_pairs_for_batch）
  3. API呼び出し: OpenAI API (client.responses.parse)が主なボトルネック
  4. 処理時間: 90分以上（全データセット処理時）

  ボトルネック箇所:
  - 884-889行目: client.responses.parse() - 同期的API呼び出し
  - 各チャンクの処理が直列実行

  🚀 Celeryを使った並列化提案

  1. アーキテクチャ設計

  [メインプロセス] → [Redis/RabbitMQ] → [Celeryワーカー群]
                           ↓
                     [タスクキュー]
                           ↓
                [並列OpenAI API呼び出し]

  2. 並列化可能な処理単位

  | レベル     | 対象関数/クラス                    | 並列化方法                    |
  |---------|-----------------------------|--------------------------|
  | チャンクレベル | generate_qa_pairs_for_chunk | 各チャンクを独立したCeleryタスクとして実行 |
  | バッチレベル  | generate_qa_pairs_for_batch | バッチ内の各チャンクを並列処理          |
  | 文書レベル   | generate_qa_for_dataset     | 複数文書を同時処理                |

  3. Celeryタスク設計

  # celery_tasks.py（新規作成）
  @celery_app.task(bind=True, max_retries=3)
  def process_chunk_task(self, chunk_data, config, model):
      """単一チャンクのQ/A生成タスク"""

  @celery_app.task
  def process_batch_task(chunks_batch, config, model):
      """バッチ処理タスク"""

  @celery_app.task
  def aggregate_results_task(qa_pairs_list):
      """結果集約タスク"""

  4. 実装時の考慮事項

  | 項目        | 推奨設定    | 理由                 |
  |-----------|---------|--------------------|
  | ワーカー数     | 5-10    | OpenAI APIレート制限を考慮 |
  | 同時実行数     | 5       | API制限とコストのバランス     |
  | タスクタイムアウト | 300秒    | 長時間処理の防止           |
  | リトライ戦略    | 指数バックオフ | APIエラー対策           |
  | キューイング    | Redis推奨 | 高速・シンプル            |

  5. レート制限対策

  # OpenAI API制限対応
  RATE_LIMITS = {
      "gpt-5-mini": {
          "rpm": 3500,  # Requests Per Minute
          "tpm": 200000  # Tokens Per Minute
      }
  }

  # Celeryでのレート制限実装
  @celery_app.task(rate_limit='50/m')  # 分あたり50リクエストに制限

  6. エラーハンドリング

  # リトライ戦略
  retry_kwargs = {
      'max_retries': 3,
      'countdown': 60,  # 60秒後にリトライ
      'retry_jitter': True,
      'retry_backoff': True,
      'retry_backoff_max': 600  # 最大10分
  }

  7. 進捗モニタリング

  | ツール        | 用途            |
  |------------|---------------|
  | Flower     | Celeryタスクの可視化 |
  | Prometheus | メトリクス収集       |
  | カスタムログ     | 進捗率・残り時間表示    |

  📈 期待される改善効果

  | 指標     | 現状      | 目標         | 改善率      |
  |--------|---------|------------|----------|
  | 処理時間   | 90分以上   | 15-20分     | 75-80%削減 |
  | スループット | 1チャンク/秒 | 5-10チャンク/秒 | 5-10倍    |
  | CPU使用率 | 10-20%  | 60-80%     | 効率向上     |

  🔧 実装ステップ

  1. 環境準備
    - Celery, Redis インストール
    - celery_config.py 作成
  2. タスク分離
    - generate_qa_pairs_for_chunk をCeleryタスクに変換
    - バッチ処理の並列化実装
  3. キュー管理
    - 優先度付きキュー設定
    - デッドレターキュー実装
  4. モニタリング
    - Flower設定
    - ログ集約システム構築
  5. テスト
    - 小規模データでの動作確認
    - 負荷テストと最適化

  ⚠️ 注意事項

  - コスト管理: 並列化でAPI呼び出し頻度が上がるため、コスト監視必須
  - データ整合性: 結果の順序保証が必要な場合は追加実装要
  - メモリ管理: 大量タスク時のメモリ使用量に注意
  - グレースフルシャットダウン: 処理中のタスクの適切な終了処理

  この提案により、90分以上かかっていた処理を15-20分程度に短縮できると期待されます。

> 1. 環境準備
    - Celery, Redis インストール
    - celery_config.py 作成

を表示せよ。

⏺ 1. 環境準備 - Celery/Redis インストールと設定

  📦 必要なパッケージのインストール

  # Celeryと関連パッケージ
  pip install celery[redis]==5.3.4
  pip install redis==5.0.1
  pip install flower==2.0.1  # モニタリング用
  pip install celery-progress==0.3  # 進捗表示用

  # 既存のrequirements.txtに追加
  echo "celery[redis]==5.3.4" >> requirements.txt
  echo "redis==5.0.1" >> requirements.txt
  echo "flower==2.0.1" >> requirements.txt
  echo "celery-progress==0.3" >> requirements.txt

  🐳 Redis サーバーのセットアップ

  Docker Composeファイル更新

  # docker-compose/docker-compose.yml に追加
  version: '3.8'

  services:
    qdrant:
      image: qdrant/qdrant
      ports:
        - "6333:6333"
      volumes:
        - qdrant_data:/qdrant/storage

    redis:
      image: redis:7-alpine
      ports:
        - "6379:6379"
      command: redis-server --appendonly yes
      volumes:
        - redis_data:/data
      healthcheck:
        test: ["CMD", "redis-cli", "ping"]
        interval: 5s
        timeout: 3s
        retries: 5

  volumes:
    qdrant_data:
    redis_data:

  Redis起動

  # Docker Composeで起動
  docker-compose -f docker-compose/docker-compose.yml up -d redis

  # または、ローカルでRedisをインストール（Mac）
  brew install redis
  brew services start redis

  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install redis-server
  sudo systemctl start redis-server

  ⚙️ celery_config.py の作成

  # celery_config.py
  """
  Celery設定ファイル
  Q/A生成タスクの並列処理用設定
  """

  import os
  from kombu import Exchange, Queue
  from celery import Celery
  from dotenv import load_dotenv

  # 環境変数読み込み
  load_dotenv()

  # Celeryアプリケーション初期化
  app = Celery('qa_generation')

  # Redis設定
  REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

  # Celery設定クラス
  class CeleryConfig:
      # ブローカー設定（Redis）
      broker_url = REDIS_URL
      result_backend = REDIS_URL

      # タスク設定
      task_serializer = 'json'
      accept_content = ['json']
      result_serializer = 'json'
      timezone = 'Asia/Tokyo'
      enable_utc = True

      # ワーカー設定
      worker_prefetch_multiplier = 1  # 各ワーカーが一度に取得するタスク数
      worker_max_tasks_per_child = 50  # メモリリーク対策
      worker_disable_rate_limits = False

      # タスク実行設定
      task_acks_late = True  # タスク完了後にACK
      task_reject_on_worker_lost = True
      task_time_limit = 300  # タスクタイムアウト（5分）
      task_soft_time_limit = 270  # ソフトタイムアウト（4.5分）

      # レート制限（OpenAI API制限対応）
      task_annotations = {
          'tasks.process_chunk_task': {
              'rate_limit': '50/m',  # 分あたり50リクエスト
          },
          'tasks.process_batch_task': {
              'rate_limit': '10/m',  # 分あたり10バッチ
          }
      }

      # キュー設定
      task_routes = {
          'tasks.process_chunk_task': 'high_priority',
          'tasks.process_batch_task': 'normal_priority',
          'tasks.aggregate_results_task': 'low_priority',
      }

      # キュー定義
      task_queues = (
          Queue('high_priority', Exchange('high_priority'), routing_key='high'),
          Queue('normal_priority', Exchange('normal_priority'), routing_key='normal'),
          Queue('low_priority', Exchange('low_priority'), routing_key='low'),
      )

      # リトライ設定
      task_autoretry_for = (Exception,)
      task_retry_kwargs = {
          'max_retries': 3,
          'countdown': 60,  # 60秒後にリトライ
          'retry_jitter': True,  # ジッター追加
      }

      # 結果の有効期限
      result_expires = 3600  # 1時間

      # Celery Beat設定（定期タスク用）
      beat_schedule = {
          'cleanup-old-results': {
              'task': 'tasks.cleanup_old_results',
              'schedule': 3600.0,  # 1時間ごと
          },
      }

  # 設定を適用
  app.config_from_object(CeleryConfig())

  # OpenAI API設定
  OPENAI_CONFIG = {
      'api_key': os.getenv('OPENAI_API_KEY'),
      'models': {
          'gpt-5-mini': {
              'rpm_limit': 3500,  # Requests Per Minute
              'tpm_limit': 200000,  # Tokens Per Minute
              'max_retries': 3,
              'retry_delay': 60
          },
          'gpt-4': {
              'rpm_limit': 500,
              'tpm_limit': 40000,
              'max_retries': 3,
              'retry_delay': 120
          }
      }
  }

  # ログ設定
  LOGGING_CONFIG = {
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
          'default': {
              'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          },
      },
      'handlers': {
          'console': {
              'class': 'logging.StreamHandler',
              'formatter': 'default',
          },
          'file': {
              'class': 'logging.FileHandler',
              'filename': 'logs/celery_qa_generation.log',
              'formatter': 'default',
          },
      },
      'loggers': {
          'celery': {
              'handlers': ['console', 'file'],
              'level': 'INFO',
          },
          'tasks': {
              'handlers': ['console', 'file'],
              'level': 'DEBUG',
          },
      },
  }

  # エクスポート
  __all__ = ['app', 'CeleryConfig', 'OPENAI_CONFIG', 'LOGGING_CONFIG']

  if __name__ == '__main__':
      # 設定確認用
      print("Celery Configuration:")
      print(f"Broker URL: {CeleryConfig.broker_url}")
      print(f"Result Backend: {CeleryConfig.result_backend}")
      print(f"Task Time Limit: {CeleryConfig.task_time_limit}s")
      print("\nOpenAI Configuration:")
      for model, config in OPENAI_CONFIG['models'].items():
          print(f"  {model}: RPM={config['rpm_limit']}, TPM={config['tpm_limit']}")

  🔐 環境変数設定（.env）

  # .envファイルに追加
  REDIS_URL=redis://localhost:6379/0
  CELERY_BROKER_URL=redis://localhost:6379/0
  CELERY_RESULT_BACKEND=redis://localhost:6379/0

  # Flower（モニタリング）設定
  FLOWER_PORT=5555
  FLOWER_BASIC_AUTH=admin:password  # 本番環境では強力なパスワードに変更

  🚀 Celeryワーカー起動スクリプト

  # start_workers.sh
  #!/bin/bash

  # Celeryワーカー起動スクリプト
  echo "Starting Celery workers..."

  # ログディレクトリ作成
  mkdir -p logs

  # 高優先度ワーカー（3プロセス）
  celery -A celery_config worker \
      --loglevel=info \
      --concurrency=3 \
      --hostname=worker_high@%h \
      --queues=high_priority \
      --logfile=logs/worker_high.log &

  # 通常優先度ワーカー（2プロセス）
  celery -A celery_config worker \
      --loglevel=info \
      --concurrency=2 \
      --hostname=worker_normal@%h \
      --queues=normal_priority \
      --logfile=logs/worker_normal.log &

  # 低優先度ワーカー（1プロセス）
  celery -A celery_config worker \
      --loglevel=info \
      --concurrency=1 \
      --hostname=worker_low@%h \
      --queues=low_priority \
      --logfile=logs/worker_low.log &

  # Flower（モニタリング）起動
  celery -A celery_config flower \
      --port=${FLOWER_PORT:-5555} \
      --basic_auth=${FLOWER_BASIC_AUTH:-admin:password} &

  echo "Celery workers started!"
  echo "Monitor at: http://localhost:5555"

  📊 動作確認

  # Redis接続確認
  redis-cli ping
  # 期待される出力: PONG

  # Celeryワーカー起動
  chmod +x start_workers.sh
  ./start_workers.sh

  # ワーカーステータス確認
  celery -A celery_config status

  # Flowerでモニタリング
  # ブラウザで http://localhost:5555 を開く

  🔍 トラブルシューティング

  # Redisが起動しない場合
  sudo systemctl status redis-server
  sudo journalctl -u redis-server -n 50

  # Celeryワーカーが接続できない場合
  celery -A celery_config inspect ping

  # ログ確認
  tail -f logs/worker_*.log

  # プロセス確認
  ps aux | grep celery

