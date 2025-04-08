"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
import pandas as pd


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # CSVファイルとそれ以外のファイルを分離
    csv_docs = [doc for doc in docs_all if doc.metadata.get("source", "").endswith(".csv")]
    other_docs = [doc for doc in docs_all if not doc.metadata.get("source", "").endswith(".csv")]
    
    # 社員名簿ファイルを検出
    detect_employee_csv_file(csv_docs)
    
    # CSVファイルを特別に処理
    processed_csv_docs = process_csv_documents(csv_docs)
    
    # 通常のテキスト文書用のチャンク分割処理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    # 通常のテキスト文書をチャンク分割
    splitted_other_docs = text_splitter.split_documents(other_docs)
    
    # 全ドキュメントを結合
    all_processed_docs = splitted_other_docs + processed_csv_docs
    
    # ベクターストアの作成
    db = Chroma.from_documents(all_processed_docs, embedding=embeddings)
    
    # セッション変数にベクターストアを保存
    st.session_state.vectorstore = db
    
    # ベクターストアを検索するRetrieverの作成（デフォルト値）
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_DOCUMENT_COUNT_DEFAULT})
    
    logger.info(f"Retrieverを作成しました。総ドキュメント数: {len(all_processed_docs)}")
    if "employee_csv_path" in st.session_state:
        logger.info(f"社員名簿ファイルを検出しました: {st.session_state.employee_csv_path}")


def detect_employee_csv_file(csv_docs):
    """
    社員名簿ファイルをCSVドキュメントリストから検出し、最も信頼度の高いものをセッション変数に保存する
    
    Args:
        csv_docs: CSVドキュメントのリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    if not csv_docs:
        logger.info("CSVファイルが見つかりませんでした。")
        return
    
    # 候補リストと信頼度スコア
    candidates = []
    
    for doc in csv_docs:
        source = doc.metadata.get("source", "")
        
        try:
            # ファイル名に社員名簿らしい文字列が含まれているか確認
            filename_score = 0
            for keyword in ["社員", "名簿", "employee", "staff", "member"]:
                if keyword in source.lower():
                    filename_score += 2
            
            # CSVファイルの内容を確認
            df = pd.read_csv(source)
            
            # 行数が多いほど信頼度アップ（社員名簿は通常複数行）
            row_score = min(5, len(df) / 10)  # 最大5点
            
            # 特徴的なカラム名を確認
            column_score = 0
            expected_columns = ["名前", "氏名", "社員", "部署", "所属", "役職", "スキル", "入社"]
            for col in df.columns:
                for keyword in expected_columns:
                    if keyword in col:
                        column_score += 1
                        break
            
            # 総合スコアの計算
            total_score = filename_score + row_score + column_score
            
            # カラム構成からさらに判定
            has_name_col = any(keyword in col for col in df.columns for keyword in ["名前", "氏名", "社員"])
            has_dept_col = any(keyword in col for col in df.columns for keyword in ["部署", "所属"])
            
            # 名前と部署のカラムがある場合は特に信頼度が高い
            if has_name_col and has_dept_col:
                total_score += 5
            
            # 候補リストに追加
            candidates.append({
                "path": source,
                "score": total_score,
                "columns": list(df.columns),
                "rows": len(df)
            })
            
            logger.info(f"CSVファイル評価: {source}, スコア: {total_score}")
            
        except Exception as e:
            logger.error(f"CSVファイル評価エラー: {source} - {str(e)}")
    
    # 信頼度でソートして最も高いものを選択
    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidate = candidates[0]
        
        # 最低信頼度のしきい値を設定
        if top_candidate["score"] >= 5:  # 最低5点以上が社員名簿と判断
            # セッション変数に保存
            st.session_state.employee_csv_path = top_candidate["path"]
            st.session_state.employee_csv_info = top_candidate
            logger.info(f"社員名簿として検出: {top_candidate['path']}, スコア: {top_candidate['score']}")
        else:
            logger.info(f"社員名簿と判断できるCSVファイルが見つかりませんでした。最高スコア: {top_candidate['score']}")
    else:
        logger.info("評価可能なCSVファイルが見つかりませんでした。")


def process_csv_documents(csv_docs):
    """
    CSVドキュメントを特別に処理する
    
    Args:
        csv_docs: CSVドキュメントのリスト
        
    Returns:
        処理済みのCSVドキュメントのリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    processed_docs = []
    
    for doc in csv_docs:
        source = doc.metadata.get("source", "")
        
        try:
            # 社員名簿として検出されたファイルか確認
            is_employee_csv = "employee_csv_path" in st.session_state and source == st.session_state.employee_csv_path
            
            # 社員名簿の場合は特別処理
            if is_employee_csv:
                logger.info(f"社員名簿CSVを特別処理: {source}")
                
                # CSVをDataFrameとして読み込み
                df = pd.read_csv(source)
                
                # 部署カラムの特定
                dept_col = None
                for col in df.columns:
                    if any(keyword in col for keyword in ["部署", "所属"]):
                        dept_col = col
                        break
                
                # 部署ごとにグループ化して処理
                if dept_col:
                    for dept, group in df.groupby(dept_col):
                        # 部署別のデータを作成
                        content = f"部署「{dept}」の社員情報:\n{group.to_string(index=False)}"
                        processed_docs.append(
                            Document(
                                page_content=content,
                                metadata={"source": source, "department": dept, "is_employee_data": True}
                            )
                        )
                    
                    # 全体データも追加
                    all_content = f"全社員データ:\n{df.to_string(index=False)}"
                    processed_docs.append(
                        Document(
                            page_content=all_content,
                            metadata={"source": source, "type": "全社員一覧", "is_employee_data": True}
                        )
                    )
                else:
                    # 部署カラムがない場合は全体を一つのドキュメントに
                    content = f"社員データ:\n{df.to_string(index=False)}"
                    processed_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": source, "is_employee_data": True}
                        )
                    )
            else:
                # 通常のCSVは行ごとに処理
                processed_docs.append(doc)
        
        except Exception as e:
            logger.error(f"CSVファイル処理エラー: {source} - {str(e)}")
            processed_docs.append(doc)  # エラーの場合は元のドキュメントを使用
    
    return processed_docs


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []
    
    # 追加: モードの初期設定を行う
    if "mode" not in st.session_state:
        st.session_state.mode = ct.ANSWER_MODE_1  # 初期モードとして ANSWER_MODE_1 を設定


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s