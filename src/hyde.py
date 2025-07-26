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

from langsmith import traceable

# ファイルフィルター関数を定義（.mdx拡張子のファイルのみを対象とする）
def file_filter(file_path: str) -> bool:
    # ファイルパスが.mdxで終わるかをチェックして真偽値を返す
    return file_path.endswith(".mdx")

# GitLoaderオブジェクトを作成（LangChainの公式リポジトリからドキュメントを読み込む設定）
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain.git",  # 対象のGitリポジトリURL
    repo_path="./langchain",  # ローカルにクローンする際の保存先パス
    branch="master",  # 取得するブランチ名
    file_filter=file_filter,  # 上記で定義したファイルフィルター関数を適用
)

# Gitリポジトリからドキュメントを実際に読み込む（時間がかかる処理）
documents = loader.load()
# 読み込んだドキュメントの数をコンソールに出力
print(len(documents))

# OpenAIの埋め込みモデル（text-embedding-3-small）のインスタンスを作成
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 読み込んだドキュメントから埋め込みベクトルを生成し、Chromaデータベースを作成
db = Chroma.from_documents(documents, embeddings)

# RAG用のプロンプトテンプレートを定義（文脈と質問を受け取って回答を生成する形式）
hypotheical_prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
文脈にない情報については「分からない」と答えてください。

<question>
{question}
</question>

''')

# ChatOpenAIモデル（GPT-4o-mini）のインスタンスを作成（temperature=0で決定的な出力）
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 仮説生成チェーンを構築
hypotheical_chain = hypotheical_prompt | model | StrOutputParser()
# Chromaデータベースからリトリーバー（関連文書を検索するオブジェクト）を作成
retriever = db.as_retriever()

# 複数のドキュメントを1つの文字列に結合する関数を定義
def format_docs(docs):
    # 各ドキュメントのpage_contentを取得し、2つの改行で結合して返す
    return "\n\n".join(doc.page_content for doc in docs)

# RAGチェーンを構築（質問→関連文書検索→プロンプト生成→LLM→回答の流れ）
hyde_rag_chain = (
    { # 質問に関連する文書を検索し、format_docs関数で整形
        "question": hypotheical_chain | retriever | format_docs
    }
    | prompt  # 上記の結果をプロンプトテンプレートに代入
    | model  # プロンプトをChatOpenAIモデルに送信
    | StrOutputParser()  # モデルの出力を文字列として取得
)

# 構築したRAGチェーンに質問を投げかけて結果を出力
print(hyde_rag_chain.invoke("LangChainの概要を教えて"))