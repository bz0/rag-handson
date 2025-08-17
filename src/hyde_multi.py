# HyDE（Hypothetical Document Embeddings）を使ったRAGシステムの実装
# HyDEは質問から仮説文書を生成し、その仮説文書で検索することで検索精度を向上させる手法

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

# LangSmithのトレーシング機能をインポート（実行過程を可視化・監視するため）
from langsmith import traceable

# ファイルフィルター関数を定義（.mdx拡張子のファイルのみを対象とする）
def file_filter(file_path: str) -> bool:
    # ファイルパスが.mdxで終わるかをチェックして真偽値を返す
    # LangChainのドキュメントは.mdx形式で書かれているため
    return file_path.endswith(".mdx")

# GitLoaderオブジェクトを作成（LangChainの公式リポジトリからドキュメントを読み込む設定）
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain.git",  # 対象のGitリポジトリURL
    repo_path="./langchain",  # ローカルにクローンする際の保存先パス
    branch="master",  # 取得するブランチ名（最新の開発版）
    file_filter=file_filter,  # 上記で定義したファイルフィルター関数を適用
)

# Gitリポジトリからドキュメントを実際に読み込む（初回実行時は時間がかかる処理）
documents = loader.load()
# 読み込んだドキュメントの数をコンソールに出力（処理の進行状況を確認するため）
print(f"読み込んだドキュメント数: {len(documents)}")

# OpenAIの埋め込みモデル（text-embedding-3-small）のインスタンスを作成
# このモデルはテキストを1536次元のベクトルに変換する
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 読み込んだドキュメントから埋め込みベクトルを生成し、Chromaデータベースを作成
# この処理により各ドキュメントがベクトル化されて検索可能になる
db = Chroma.from_documents(documents, embeddings)

# Multi-HyDE用の複数の仮説文書生成プロンプトテンプレートを定義
# 異なる視点から仮説文書を生成することで、より幅広い関連文書を検索できる

# 1. 一般的な説明に焦点を当てた仮説文書生成プロンプト
general_prompt = ChatPromptTemplate.from_template('''
質問に対して、一般的で包括的な説明が含まれていそうな仮想的な文書の内容を書いてください。
概要や基本的な説明を含む文書を想像して書いてください。

質問: {question}

仮想的な文書（一般説明版）:
''')

# 2. 技術的な詳細に焦点を当てた仮説文書生成プロンプト
technical_prompt = ChatPromptTemplate.from_template('''
質問に対して、技術的な詳細や実装方法が含まれていそうな仮想的な文書の内容を書いてください。
コードサンプルや詳細な技術仕様を含む文書を想像して書いてください。

質問: {question}

仮想的な文書（技術詳細版）:
''')

# 3. 使用例や実践的な内容に焦点を当てた仮説文書生成プロンプト
practical_prompt = ChatPromptTemplate.from_template('''
質問に対して、具体的な使用例や実践的なガイドが含まれていそうな仮想的な文書の内容を書いてください。
チュートリアルや実際の使い方を含む文書を想像して書いてください。

質問: {question}

仮想的な文書（実用例版）:
''')

# ChatOpenAIモデル（GPT-4o-mini）のインスタンスを作成
# temperature=0.7で少し創造性を持たせる（仮説文書生成には創造性が重要）
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 複数の仮説文書生成チェーンを構築
# 各プロンプトから異なる視点の仮説文書を生成する
general_chain = general_prompt | model | StrOutputParser()
technical_chain = technical_prompt | model | StrOutputParser()
practical_chain = practical_prompt | model | StrOutputParser()

# 複数の仮説文書を並行生成する関数
def generate_multiple_hypotheses(question: str) -> list[str]:
    """複数の異なる視点から仮説文書を生成する
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        3つの異なる仮説文書のリスト
    """
    # 3つの異なるチェーンを並行実行して仮説文書を生成
    # 並行実行により処理時間を短縮できる
    general_hyp = general_chain.invoke({"question": question})
    technical_hyp = technical_chain.invoke({"question": question})
    practical_hyp = practical_chain.invoke({"question": question})
    
    return [general_hyp, technical_hyp, practical_hyp]

# Chromaデータベースからリトリーバー（関連文書を検索するオブジェクト）を作成
# search_kwargs={"k": 5}で上位5件の関連文書を取得するよう設定
retriever = db.as_retriever(search_kwargs={"k": 5})

# 複数の仮説文書を使って並行検索を実行し、結果をマージする関数
def search_with_multiple_hypotheses(question: str) -> list:
    """複数の仮説文書を使って検索を実行し、結果をマージする
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        マージされた検索結果のドキュメントリスト
    """
    # 複数の仮説文書を生成
    hypotheses = generate_multiple_hypotheses(question)
    
    # 各仮説文書で検索を実行
    all_results = []
    doc_scores = {}  # ドキュメントの重複回数を記録（スコアとして使用）
    
    for i, hypothesis in enumerate(hypotheses):
        # 各仮説文書で検索を実行（上位3件ずつ取得）
        results = retriever.invoke(hypothesis)
        
        for doc in results:
            # ドキュメントのIDを生成（内容のハッシュを使用）
            doc_id = hash(doc.page_content)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            
            # 複数の仮説で見つかったドキュメントほど高スコア
            doc_scores[doc_id]["score"] += 1
    
    # スコア順でソートして上位5件を返す
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:5]]

# 複数のドキュメントを1つの文字列に結合する関数を定義
def format_docs(docs):
    # 各ドキュメントのpage_contentを取得し、2つの改行で結合して返す
    # これにより複数の文書が読みやすい形式で統合される
    return "\n\n".join(doc.page_content for doc in docs)

# 最終回答生成用のプロンプトテンプレート
# 検索された文書（文脈）と質問を使って最終的な回答を生成する
answer_prompt = ChatPromptTemplate.from_template('''
以下の文脈を踏まえて質問に回答してください。
文脈にない情報については「分からない」と答えてください。

文脈:
{context}

質問: {question}

回答:
''')

# Multi-HyDEを使ったRAGチェーンを構築
# 複数の仮説文書を使った検索により、より包括的な文脈を取得
multi_hyde_rag_chain = (
    {
        # 元の質問をそのまま渡す（最終的な回答生成で必要）
        "question": RunnablePassthrough(),
        # Multi-HyDEの核心部分：質問→複数仮説文書生成→複数検索→結果マージ→文書整形
        "context": lambda x: format_docs(search_with_multiple_hypotheses(x))
    }
    # 文脈と質問をプロンプトテンプレートに代入
    | answer_prompt
    # プロンプトをChatOpenAIモデルに送信して回答を生成
    | model
    # モデルの出力を文字列として取得
    | StrOutputParser()
)

# 従来のシングルHyDEチェーンも比較用に保持（元のhypothetical_promptは削除されたので再定義）
single_hypothetical_prompt = ChatPromptTemplate.from_template('''
質問に対して、その答えが含まれていそうな仮想的な文書の内容を書いてください。
実際の情報を知らなくても構いません。質問の答えを含む文書がどのような内容になりそうかを想像して書いてください。

質問: {question}

仮想的な文書:
''')

single_hypothetical_chain = single_hypothetical_prompt | model | StrOutputParser()

hyde_rag_chain = (
    {
        # 元の質問をそのまま渡す（最終的な回答生成で必要）
        "question": RunnablePassthrough(),
        # 従来のHyDE：質問→単一仮説文書生成→検索→文書整形
        "context": single_hypothetical_chain | retriever | format_docs
    }
    # 文脈と質問をプロンプトテンプレートに代入
    | answer_prompt
    # プロンプトをChatOpenAIモデルに送信して回答を生成
    | model
    # モデルの出力を文字列として取得
    | StrOutputParser()
)

# Multi-HyDE RAGチェーンをトレース可能にする関数（LangSmithで実行過程を監視可能）
@traceable(name="multi_hyde_rag")
def ask_multi_hyde_rag(question: str) -> str:
    """Multi-HyDEを使ったRAGで質問に回答する関数
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        Multi-HyDEを使って生成された回答
    """
    # 構築したMulti-HyDEチェーンを実行して回答を取得
    return multi_hyde_rag_chain.invoke(question)

# HyDE RAGチェーンをトレース可能にする関数（比較用）
@traceable(name="hyde_rag")
def ask_hyde_rag(question: str) -> str:
    """従来のHyDEを使ったRAGで質問に回答する関数
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        従来のHyDEを使って生成された回答
    """
    # 構築したHyDEチェーンを実行して回答を取得
    return hyde_rag_chain.invoke(question)

# 通常のRAGチェーン（比較用）も構築
# HyDEと違い、質問を直接使ってベクトル検索を行う従来手法
normal_rag_chain = (
    {
        # 元の質問をそのまま渡す
        "question": RunnablePassthrough(),
        # 質問を直接使ってベクトル検索→文書整形（仮説文書生成なし）
        "context": retriever | format_docs
    }
    # 同じプロンプトテンプレートを使用
    | answer_prompt
    # 同じモデルで回答生成
    | model
    # 同じ出力パーサーで文字列取得
    | StrOutputParser()
)

# 通常RAGもトレース可能にする関数
@traceable(name="normal_rag")
def ask_normal_rag(question: str) -> str:
    """通常のRAGで質問に回答する関数
    
    Args:
        question: ユーザーからの質問文
        
    Returns:
        通常のRAGで生成された回答
    """
    # 構築した通常RAGチェーンを実行して回答を取得
    return normal_rag_chain.invoke(question)

# メイン実行部分（このファイルが直接実行された場合のみ動作）
if __name__ == "__main__":
    # テスト用の質問を定義
    test_question = "LangChainの概要を教えて"
    
    print("=" * 60)
    print("RAG手法の比較テスト")
    print("=" * 60)
    
    # Multi-HyDEを使ったRAGの結果を表示（新しい改良手法）
    print("\n=== Multi-HyDE RAGの結果（複数仮説文書使用） ===")
    # Multi-HyDE RAG関数を呼び出して回答を取得
    multi_hyde_result = ask_multi_hyde_rag(test_question)
    # 結果をコンソールに出力
    print(multi_hyde_result)
    
    # 従来のHyDEを使ったRAGの結果を表示（比較用）
    print("\n=== 従来HyDE RAGの結果（単一仮説文書使用） ===")
    # 従来のHyDE RAG関数を呼び出して回答を取得
    hyde_result = ask_hyde_rag(test_question)
    # 結果をコンソールに出力
    print(hyde_result)
    
    # 通常RAGの結果を表示（ベースライン比較用）
    print("\n=== 通常RAGの結果（仮説文書なし） ===")
    # 通常RAG関数を呼び出して回答を取得
    normal_result = ask_normal_rag(test_question)
    # 結果をコンソールに出力
    print(normal_result)
    
    print("\n" + "=" * 60)
    print("比較テスト完了")
    print("=" * 60)