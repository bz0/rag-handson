# LangChainのGitリポジトリローダーをインポート（Gitリポジトリからドキュメントを読み込むため）
from langchain_community.document_loaders import GitLoader
# Chromaベクトルデータベースをインポート（文書の埋め込みベクトルを保存・検索するため）
from langchain_chroma import Chroma
# OpenAIの埋め込みモデルをインポート（テキストをベクトルに変換するため）
from langchain_openai import OpenAIEmbeddings
# 文字列出力パーサーをインポート（LLMの出力を文字列として取得するため）
from langchain_core.output_parsers import StrOutputParser
# チャットプロンプトテンプレートをインポート（構造化されたプロンプトを作成するため）
from langchain_core.prompts import ChatPromptTemplate
# RunnablePassthroughをインポート（チェーン内で値をそのまま渡すため）
from langchain_core.runnables import RunnablePassthrough
# OpenAIのチャットモデルをインポート（GPTモデルを使用するため）
from langchain_openai import ChatOpenAI

# LangSmithトレーシング用のインポート（ドキュメント推奨方式）
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
import os

# LangSmithトレーシング設定の確認
print("🔍 LangSmithトレーシング設定を確認中...")
tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if tracing_enabled:
    print("✅ LangSmithトレーシングが有効です")
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        print(f"🔑 LangSmith APIキー: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("⚠️ LANGSMITH_API_KEY が設定されていません")
else:
    print("❌ LangSmithトレーシングが無効です（環境変数LANGSMITH_TRACING=trueを設定してください）")

# OpenAIクライアントをLangSmithでラップ（ドキュメント推奨方式）
openai_client = wrap_openai(OpenAI())

# ファイルフィルター関数を定義（.mdx拡張子のファイルのみを対象とする）
@traceable(name="file_filter")  # LangSmithでトレース
def file_filter(file_path: str) -> bool:
    """ファイルパスが.mdxで終わるかをチェックして真偽値を返す"""
    return file_path.endswith(".mdx")

# ドキュメント読み込み処理をトレース可能にする
@traceable(name="load_documents")
def load_documents_from_git():
    """Gitリポジトリからドキュメントを読み込む"""
    print("📦 Gitリポジトリからドキュメントを読み込み中...")
    
    # GitLoaderオブジェクトを作成
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain.git",  # 対象のGitリポジトリURL
        repo_path="./langchain",  # ローカルにクローンする際の保存先パス
        branch="master",  # 取得するブランチ名
        file_filter=file_filter,  # 上記で定義したファイルフィルター関数を適用
    )
    
    # ドキュメントを実際に読み込む
    documents = loader.load()
    print(f"✅ 読み込み完了: {len(documents)}個のドキュメント")
    
    return documents

# ベクトルデータベース作成処理をトレース可能にする
@traceable(name="create_vector_database")
def create_vector_db(documents):
    """ドキュメントからベクトルデータベースを作成"""
    print("🔢 ベクトルデータベースを作成中...")
    
    # OpenAIの埋め込みモデルを作成
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chromaデータベースを作成
    db = Chroma.from_documents(documents, embeddings)
    print("✅ ベクトルデータベース作成完了")
    
    return db, embeddings

# リトリーバー関数をトレース可能にする（ドキュメント推奨方式）
@traceable(name="retriever")
def retriever(query: str):
    """質問に関連するドキュメントを検索"""
    docs = db.as_retriever().get_relevant_documents(query)
    print(f"🔍 検索された関連ドキュメント数: {len(docs)}")
    return [doc.page_content for doc in docs]

# メインのRAG関数をトレース可能にする（ドキュメント推奨方式）
@traceable(name="rag")
def rag(question: str):
    """RAGシステムによる質問応答処理"""
    print(f"\n💭 質問: {question}")
    
    # ステップ1: 関連文書検索
    docs = retriever(question)
    
    # ステップ2: システムメッセージを構築
    system_message = """以下の文脈だけを踏まえて質問に回答してください。
文脈にない情報については「分からない」と答えてください。

{docs}""".format(docs="\n\n".join(docs))
    
    # ステップ3: OpenAI APIを呼び出し（LangSmithでラップ済み）
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
    
    answer = response.choices[0].message.content
    print(f"💡 回答: {answer}")
    return answer

# メイン処理：ドキュメント読み込みとベクトルDB作成
documents = load_documents_from_git()
db, embeddings = create_vector_db(documents)

# 複数の質問でテスト
test_questions = [
    "LangChainの概要を教えて",
    "LangChainでRAGを実装する方法は？",
    "LangChainのドキュメントローダーにはどんな種類がありますか？"
]

print("\n🚀 RAGシステムテスト開始")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n📋 テスト {i}/{len(test_questions)}")
    try:
        rag(question)
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    
    # 次の質問まで少し間隔を空ける
    if i < len(test_questions):
        print("\n" + "-" * 30)

print("\n🎉 RAGシステムテスト完了！")
print("🔗 LangSmithダッシュボードでトレースを確認してください: https://smith.langchain.com/")