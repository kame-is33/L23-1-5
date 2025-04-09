"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if isinstance(source, str) and source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    if not message:
        return ct.COMMON_ERROR_MESSAGE
        
    if isinstance(message, list):
        error_text = "\n".join(message)
    else:
        error_text = str(message)
        
    return f"{error_text}\n{ct.COMMON_ERROR_MESSAGE}"


def check_files_for_updates(file_paths, reference_time):
    """
    指定されたファイル群が reference_time より後に更新されたかどうかをチェックします

    Args:
        file_paths: チェックするファイルパスのリスト
        reference_time: 基準となる日時（datetime オブジェクト）

    Returns:
        更新されたファイル名のリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    if not file_paths or not reference_time:
        return []
        
    updated_files = []
    for path in file_paths:
        try:
            if not os.path.exists(path):
                continue
                
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            if mod_time > reference_time:
                updated_files.append(path)
                logger.info(f"ファイル更新を検知: {path}")
        except (PermissionError, OSError) as e:
            logger.warning(f"ファイル更新チェックエラー: {path} - {e}")
    
    return updated_files


def detect_special_query_type(query):
    """
    クエリが特殊処理を必要とする種類かどうかを判定する

    Args:
        query: ユーザー入力クエリ

    Returns:
        特殊クエリタイプ（"employee", "finance", "project"など）、該当しない場合はNone
    """
    if not query:
        return None
        
    for query_type, keywords in ct.SPECIAL_QUERY_PATTERNS.items():
        if any(keyword in query for keyword in keywords):
            return query_type
    
    return None


def process_employee_query(query):
    """
    従業員情報に関するクエリを特別に処理する

    Args:
        query: ユーザー入力クエリ

    Returns:
        処理結果の辞書（成功時: {"success": True, "data": データ}, 失敗時: {"success": False, "error": エラー}）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        csv_path = ct.EMPLOYEE_DATA_PATH
        
        if not os.path.exists(csv_path):
            logger.warning(f"社員情報ファイルが見つかりません: {csv_path}")
            return {"success": False, "error": "社員情報ファイルが見つかりません"}
            
        # CSVデータの読み込み
        employee_df = pd.read_csv(csv_path)
        
        # 部署カラムを特定
        dept_column = None
        for col in employee_df.columns:
            if "部署" in col or "所属" in col:
                dept_column = col
                break
        
        return {
            "success": True,
            "data": employee_df,
            "dept_column": dept_column,
            "source": csv_path
        }
        
    except Exception as e:
        logger.error(f"社員情報処理エラー: {e}")
        return {"success": False, "error": f"社員情報の処理中にエラーが発生しました: {e}"}


def validate_llm_response(llm_response):
    """
    LLMからのレスポンスが有効かどうかを検証する

    Args:
        llm_response: LLMからのレスポンス

    Returns:
        検証結果（True: 有効, False: 無効）
    """
    if not llm_response or not isinstance(llm_response, dict):
        return False
        
    # 必須キーの確認
    if "answer" not in llm_response:
        return False
        
    return True


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info(f"LLM回答取得開始: {chat_message}")
    
    # Retrieverの初期化チェック
    if "retriever" not in st.session_state:
        error_message = ct.RETRIEVER_NOT_INITIALIZED_ERROR
        logger.error(error_message)
        return {"answer": error_message, "context": []}
    
    # 特殊クエリのチェック
    query_type = detect_special_query_type(chat_message)
    modified_query = chat_message
    
    # 特殊クエリの処理
    if query_type == "employee" and st.session_state.mode == ct.ANSWER_MODE_2:
        logger.info(f"社員情報に関するクエリを検出: {chat_message}")
        result = process_employee_query(chat_message)
        
        if result["success"]:
            # 社員情報に関する特別なプロンプト追加
            modified_query = f"社員名簿を参照して次の質問に答えてください: {chat_message}"
            logger.info(f"クエリを修正: {modified_query}")
    
    # LLMのオブジェクトを用意
    try:
        llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
    except Exception as e:
        error_message = f"LLMオブジェクトの初期化に失敗しました: {e}"
        logger.error(error_message)
        return {"answer": error_message, "context": []}
    
    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
        
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    try:
        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, question_generator_prompt
        )
        
        # LLMから回答を取得する用のChainを作成
        question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
        # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # LLMへのリクエストとレスポンス取得
        llm_response = chain.invoke({"input": modified_query, "chat_history": st.session_state.chat_history})
        
        # レスポンスの検証
        if not validate_llm_response(llm_response):
            error_message = ct.INVALID_RESPONSE_ERROR
            logger.error(f"無効なLLMレスポンス: {llm_response}")
            return {"answer": error_message, "context": []}
        
        # LLMレスポンスを会話履歴に追加
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])
        
        logger.info(f"LLM回答取得完了: {llm_response['answer'][:100]}...")
        return llm_response
        
    except Exception as e:
        error_message = f"回答生成中にエラーが発生しました: {e}"
        logger.error(error_message)
        return {"answer": error_message, "context": []}