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
print(f"読み込んだドキュメント数: {len(documents)}")

# OpenAIの埋め込みモデル（text-embedding-3-small）のインスタンスを作成
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 読み込んだドキュメントから埋め込みベクトルを生成し、Chromaデータベースを作成
db = Chroma.from_documents(documents, embeddings)

# HyDE（Hypothetical Document Embeddings）用の仮説文書生成プロンプトテンプレートを定義
hypothetical_prompt = ChatPromptTemplate.from_template('''
質問に対して、その答えが含まれていそうな仮想的な文書の内容を書いてください。
実際の情報を知らなくても構いません。質問の答えを含む文書がどのような内容になりそうかを想像して書いてください。

質問: {question}

仮想的な文書:
''')

# ChatOpenAIモデル（GPT-4o-mini）のインスタンスを作成（temperature=0.7で少し創造性を持たせる）
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 仮説文書生成チェーンを構築（質問→仮説文書生成）
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# Chromaデータベースからリトリーバー（関連文書を検索するオブジェクト）を作成
retriever = db.as_retriever(search_kwargs={"k": 5})  # 上位5件の関連文書を取得

# 複数のドキュメントを1つの文字列に結合する関数を定義
def format_docs(docs):
    # 各ドキュメントのpage_contentを取得し、2つの改行で結合して返す
    return "\n\n".join(doc.page_content for doc in docs)

# 最終回答生成用のプロンプトテンプレート
answer_prompt = ChatPromptTemplate.from_template('''
以下の文脈を踏まえて質問に回答してください。
文脈にない情報については「分からない」と答えてください。

文脈:
{context}

質問: {question}

回答:
''')

# HyDEを使ったRAGチェーンを構築
# 1. 質問から仮説文書を生成
# 2. 仮説文書を使って関連文書を検索
# 3. 検索した文書を使って最終回答を生成
hyde_rag_chain = (
    {
        # 元の質問をそのまま渡す
        "question": RunnablePassthrough(),
        # 質問→仮説文書生成→仮説文書で検索→文書整形
        "context": hypothetical_chain | retriever | format_docs
    }
    | answer_prompt  # 文脈と質問をプロンプトテンプレートに代入
    | model  # プロンプトをChatOpenAIモデルに送信
    | StrOutputParser()  # モデルの出力を文字列として取得
)

# HyDE RAGチェーンをトレース可能にする
@traceable(name="hyde_rag")
def ask_hyde_rag(question: str) -> str:
    """HyDEを使ったRAGで質問に回答する関数"""
    return hyde_rag_chain.invoke(question)

# 通常のRAGチェーン（比較用）も構築
normal_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs
    }
    | answer_prompt
    | model
    | StrOutputParser()
)

@traceable(name="normal_rag")
def ask_normal_rag(question: str) -> str:
    """通常のRAGで質問に回答する関数"""
    return normal_rag_chain.invoke(question)

# テスト実行
if __name__ == "__main__":
    test_question = "LangChainの概要を教えて"
    
    print("=== HyDE RAGの結果 ===")
    hyde_result = ask_hyde_rag(test_question)
    print(hyde_result)
    
    print("\n=== 通常RAGの結果 ===")
    normal_result = ask_normal_rag(test_question)
    print(normal_result)