"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import pandas as pd
import os
import logging
import streamlit as st
import time
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_source_icon(file_path):
    """
    参照元のファイルパスに応じて、適切なアイコンを取得する

    Args:
        file_path: 参照元のファイルパス

    Returns:
        適切なアイコン
    """
    # URLの特徴を検出
    if file_path.startswith("http") or file_path.startswith("www"):
        return ct.LINK_SOURCE_ICON
    else:
        return ct.DOC_SOURCE_ICON


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 社員情報関連のキーワードを検出
    employee_keywords = ["人事", "従業員", "社員", "部署", "スキル"]
    is_employee_query = any(keyword in chat_message for keyword in employee_keywords)
    
    # 社員情報の追加処理
    if is_employee_query and st.session_state.mode == ct.ANSWER_MODE_2:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            if os.path.exists(csv_path):
                # 社員データの読み込み
                employee_df = pd.read_csv(csv_path)
                
                # 人事部のフィルタリング
                dept_column = None
                for col in employee_df.columns:
                    if "部署" in col or "所属" in col:
                        dept_column = col
                        break
                
                # 強制的に回答を生成
                from langchain_core.documents import Document
                dummy_doc = Document(
                    page_content="社員情報があります",
                    metadata={"source": "./data/社員について/社員名簿.csv"}
                )
                
                # チャットメッセージをモディファイ
                chat_message = f"社員名簿を参照して次の質問に答えてください: {chat_message}"
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報の読み込みに失敗しました: {e}")

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

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})

    # 社員情報関連の質問で「情報が見つからなかった」場合は直接問い合わせ
    if is_employee_query and llm_response["answer"] == ct.INQUIRY_NO_MATCH_ANSWER:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            columns, _ = analyze_csv_structure(csv_path)
            if columns:
                fallback_answer = f"社員名簿には以下のカラムが含まれています: {', '.join(columns)}"
            else:
                fallback_answer = "社員情報が確認できませんでした。"
            llm_response["answer"] = fallback_answer
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報による直接回答の生成に失敗しました: {e}")

    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def analyze_csv_structure(csv_path):
    """CSVファイルの構造を分析し、重要なカラム名を特定する関数"""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 全カラム名とそのデータ型を返す例
            return df.columns.tolist(), df.dtypes.to_dict()
        else:
            return None, None
    except Exception as e:
        logging.getLogger(ct.LOGGER_NAME).error(f"CSV構造分析エラー: {e}")
        return None, None


def check_files_for_updates():
    """
    ファイルの更新を検知する関数
    
    Returns:
        更新があったかどうかのブール値
    """
    # 本来はファイルの更新日時を比較するなど実装が必要だが、ここではサンプル実装
    return False