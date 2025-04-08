import pandas as pd  # この行を追加して pandas をインポート
import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

import os
from datetime import datetime

def check_files_for_updates(file_paths, reference_time):
    """
    指定されたファイル群が reference_time より後に更新されたかどうかをチェックします。
    :param file_paths: チェックするファイルパスのリスト
    :param reference_time: datetime オブジェクト
    :return: 更新されたファイル名のリスト
    """
    updated_files = []
    for path in file_paths:
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            if mod_time > reference_time:
                updated_files.append(path)
        except FileNotFoundError:
            continue  # 存在しないファイルはスキップ
    return updated_files

def build_error_message(errors):
    """
    エラーメッセージのリストまたは文字列から、フォーマット済み文字列を生成します。
    """
    if not errors:
        return "エラーは発生していません。"

    # 単一の文字列が渡された場合、リストに変換する
    if isinstance(errors, str):
        errors = [errors]

    message = "以下のエラーが発生しました：\n"
    for i, error in enumerate(errors, 1):
        message += f"{i}. {error}\n"
    return message.strip()

def analyze_csv_structure(csv_path):
    """CSVファイルの構造を分析し、重要なカラム名を特定する関数"""
    try:
        import pandas as pd
        if os.path.exists(csv_path):
            # CSVファイル読み込み・分析処理（前回提案したコード）
            ...

    except Exception as e:
        logging.getLogger(ct.LOGGER_NAME).error(f"CSV構造分析エラー: {e}")
        return None, None

def load_csv_to_vectorstore(csv_path, vectorstore):
    """CSVファイルをベクターストアに登録する関数"""
    try:
        df = pd.read_csv(csv_path)
        documents = []
        
        for index, row in df.iterrows():
            # 各行を構造化テキストに変換
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            
            # ドキュメントとして登録
            from langchain_core.documents import Document
            doc = Document(
                page_content=content,
                metadata={
                    "source": csv_path,
                    "row": index,
                    "type": "社員情報"
                }
            )
            documents.append(doc)
        
        # ベクターストアに追加
        vectorstore.add_documents(documents)
        logging.getLogger(ct.LOGGER_NAME).info(f"CSVから{len(documents)}件のドキュメントを登録しました: {csv_path}")
        
    except Exception as e:
        logging.getLogger(ct.LOGGER_NAME).error(f"CSVのベクターストア登録エラー: {e}")

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
            ("human", "{user_input}")
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
            ("human", "{user_input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # Retrieverを使って文脈情報を取得
    retrieved_docs = history_aware_retriever.invoke({
        "chat_history": st.session_state.chat_history,
        "user_input": chat_message
    })

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({
        "user_input": chat_message,
        "chat_history": st.session_state.chat_history,
        "context": retrieved_docs
    })

    # 社員情報関連の質問で「情報が見つからなかった」場合は直接問い合わせ
    if is_employee_query and llm_response["answer"] == ct.INQUIRY_NO_MATCH_ANSWER:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            # CSVの構造を分析（前回提案したフィルタリングコード）
            ...

        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報による直接回答の生成に失敗しました: {e}")

    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response

def get_source_icon(filename: str) -> str:
    """
    ファイル拡張子に応じた Material アイコンを返す。
    現時点ではすべて同一のアイコン（:material/description:）を返すが、
    拡張子に応じて異なるアイコンを割り当てられるよう構造を維持している。
📌 解説
	•	:material/picture_as_pdf: → PDF アイコン
	•	:material/article: → Word ドキュメントに近いアイコン
	•	:material/grid_on: → 表形式データ（CSV/Excel）
	•	:material/notes: → テキストファイル
	•	:material/insert_drive_file: → その他の一般的ファイル
    """
    # 今後の拡張に備えた分岐構造（現在は全て同じアイコン）
    if filename.endswith((".pdf", ".docx", ".doc", ".xlsx", ".csv", ".txt")):
        return ":material/description:"
    else:
        return ":material/description:"